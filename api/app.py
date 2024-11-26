import asyncio
import os
import signal
import concurrent.futures
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Callable
from typing import List
import logging
import nest_asyncio

import click
import uvicorn
from quart import jsonify, request, make_response
from pydantic import BaseModel
import aiofiles
import aiofiles.os
from dotenv import load_dotenv, dotenv_values, set_key, unset_key

import pandas as pd
import yaml
from quart import Quart
from quart_cors import cors  # Import quart_cors to enable CORS
from quart_uploads import UploadSet, configure_uploads

from src.auth import require_auth
from src.evaluate_history import get_new_trial_dir
from src.run import (
    run_start_trial,
    run_dashboard,
    run_chat,
)
from src.schema import (
    ChunkRequest,
    EnvVariableRequest,
    Project,
    Task,
    Status,
    TaskType,
    Trial,
    TrialConfig,
)

from src.validate import project_exists, trial_exists
from database.project_db import SQLiteProjectDB  # 올바른 임포트로 변경
from tasks.trial_tasks import (
    generate_qa_documents,
    parse_documents,
    chunk_documents,
)  # 수정된 임포트
from celery.result import AsyncResult

# uvloop을 사용하지 않도록 설정
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# 그 다음에 nest_asyncio 적용
nest_asyncio.apply()

app = Quart(__name__)

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# CORS 설정
app = cors(
    app,
    allow_origin=["http://localhost:3000"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=True,
    max_age=3600,
)

print("CORS enabled for http://localhost:3000")

# Global variables to manage tasks
tasks = {}  # task_id -> task_info # This will be the temporal DB for task infos
task_futures = {}  # task_id -> future (for forceful termination)
task_queue = asyncio.Queue()
current_task_id = None  # ID of the currently running task
lock = asyncio.Lock()  # To manage access to shared variables

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ENV = os.getenv("AUTORAG_API_ENV", "dev")
if ENV == "dev":
    WORK_DIR = os.path.join(ROOT_DIR, "../projects")
else:  # production
    WORK_DIR = os.path.join(ROOT_DIR, "projects")
if "AUTORAG_WORK_DIR" in os.environ:
    WORK_DIR = os.getenv("AUTORAG_WORK_DIR")

if "AUTORAG_WORK_DIR" in os.environ:
    WORK_DIR = os.path.join(ROOT_DIR, os.getenv("AUTORAG_WORK_DIR"))

ENV_FILEPATH = os.path.join(ROOT_DIR, f".env.{ENV}")
# 환경에 따른 WORK_DIR 설정

load_dotenv(ENV_FILEPATH)

print(f"ENV_FILEPATH: {ENV_FILEPATH}")
print(f"WORK_DIR: {WORK_DIR}")
print(f"AUTORAG_API_ENV: {ENV}")

print("--------------------------------")
print("### Server start")
print("--------------------------------")


# Ensure CORS headers are present in every response
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin == "http://localhost:3000":
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, DELETE, PATCH, OPTIONS"
        )
    return response


# Handle OPTIONS requests explicitly
@app.route("/", methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
async def options_handler(path=""):
    response = await make_response("")
    origin = request.headers.get("Origin")
    if origin == "http://localhost:3000":
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, DELETE, PATCH, OPTIONS"
        )
    return response


async def create_task(task_id: str, task: Task, func: Callable, *args) -> None:
    tasks[task_id] = {
        "function": func,
        "args": args,
        "error": None,
        "task": task,
    }
    await task_queue.put(task_id)


async def run_background_task(task_id: str, func, *args):
    """백그라운드 작업을 실행하는 함수"""
    task_info = tasks[task_id]
    task = task_info["task"]

    try:
        loop = asyncio.get_event_loop()
        logger.info(f"Executing {func.__name__} with args: {args}")

        def execute():
            return func(*args)  # 인자를 그대로 언패킹하여 전달

        result = await loop.run_in_executor(None, execute)
        task.status = Status.COMPLETED
        return result
    except Exception as e:
        logger.error(f"Task {task_id} failed with error: {func.__name__}({args}) {e}")
        task.status = Status.FAILED
        task.error = str(e)
        raise


async def task_runner():
    global current_task_id
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ProcessPoolExecutor()
    try:
        while True:
            task_id = await task_queue.get()
            async with lock:
                current_task_id = task_id
                tasks[task_id]["task"].status = Status.IN_PROGRESS

            try:
                # Get function and arguments from task info
                func = tasks[task_id]["function"]
                args = tasks[task_id].get("args", ())

                print(f"args: {args}")
                print(f"func: {func}")

                # Load env variable before running a task
                load_dotenv(ENV_FILEPATH)

                # Run the function in a separate process
                future = loop.run_in_executor(
                    executor,
                    func,
                    *args,
                )
                task_futures[task_id] = future

                await future
                if func.__name__ == run_dashboard.__name__:
                    tasks[task_id]["report_pid"] = future.result()
                elif func.__name__ == run_chat.__name__:
                    tasks[task_id]["chat_pid"] = future.result()

                # Update status on completion
                async with lock:
                    print(f"Task {task_id} is completed")
                    tasks[task_id]["task"].status = Status.COMPLETED
                    current_task_id = None
            except asyncio.CancelledError:
                tasks[task_id]["task"].status = Status.TERMINATED
                print(f"Task {task_id} has been forcefully terminated.")
            except Exception as e:
                # Handle errors
                async with lock:
                    tasks[task_id]["task"].status = Status.FAILED
                    tasks[task_id]["error"] = str(e)
                    current_task_id = None
                print(f"Task {task_id} failed with error: task_runner {e}")
                print(e)

            finally:
                task_queue.task_done()
                task_futures.pop(task_id, None)
    finally:
        executor.shutdown()


async def cancel_task(task_id: str) -> None:
    async with lock:
        future = task_futures.get(task_id)
        if future and not future.done():
            try:
                # Attempt to kill the associated process directly
                future.cancel()
            except Exception as e:
                tasks[task_id]["task"].status = Status.FAILED
                tasks[task_id]["error"] = f"Failed to terminate: {str(e)}"
                print(f"Task {task_id} failed to terminate with error: {e}")
        else:
            print(f"Task {task_id} is not running or already completed.")


@app.before_serving
async def startup():
    # Start the background task when the app starts
    app.add_background_task(task_runner)


# Project creation endpoint
@app.route("/projects", methods=["POST"])
@require_auth()
async def create_project():
    data = await request.get_json()

    # Validate required fields
    if not data or "name" not in data:
        return jsonify({"error": "Name is required"}), 400

    description = data.get("description", "")
    print(f"Set WORK_DIR environment variable to: {os.environ['AUTORAG_WORK_DIR']}")
    WORK_DIR = os.environ["AUTORAG_WORK_DIR"]
    # Create a new project
    new_project_dir = os.path.join(WORK_DIR, data["name"])
    if not os.path.exists(new_project_dir):
        os.makedirs(new_project_dir)
        os.makedirs(os.path.join(new_project_dir, "parse"))
        os.makedirs(os.path.join(new_project_dir, "chunk"))
        os.makedirs(os.path.join(new_project_dir, "qa"))
        os.makedirs(os.path.join(new_project_dir, "project"))
        os.makedirs(os.path.join(new_project_dir, "config"))
        os.makedirs(os.path.join(new_project_dir, "raw_data"))
        # SQLiteProjectDB 인스턴스 생성
        _ = SQLiteProjectDB(data["name"])
    else:
        return jsonify({"error": f'Project name already exists: {data["name"]}'}), 400

    # save at 'description.txt' file
    with open(os.path.join(new_project_dir, "description.txt"), "w") as f:
        f.write(description)

    response = Project(
        id=data["name"],
        name=data["name"],
        description=description,
        created_at=datetime.now(tz=timezone.utc),
        status="active",
        metadata={},
    )
    return jsonify(response.model_dump()), 201


async def get_project_directories():
    """Get all project directories from WORK_DIR."""
    directories = []

    # List all directories in WORK_DIR
    for item in Path(WORK_DIR).iterdir():
        if item.is_dir():
            directories.append(
                {
                    "name": item.name,
                    "status": "active",  # All projects are currently active
                    "path": str(item),
                    "last_modified_datetime": datetime.fromtimestamp(
                        item.stat().st_mtime,
                        tz=timezone.utc,
                    ),
                    "created_datetime": datetime.fromtimestamp(
                        item.stat().st_ctime,
                        tz=timezone.utc,
                    ),
                }
            )

    directories.sort(key=lambda x: x["last_modified_datetime"], reverse=True)
    return directories


@app.route("/projects", methods=["GET"])
async def list_projects():
    """List all projects with pagination. It returns the last modified projects first."""
    # Get query parameters with defaults
    page = request.args.get("page", 1, type=int)
    limit = request.args.get("limit", 10, type=int)
    status = request.args.get("status", "active")

    # Validate pagination parameters
    if page < 1:
        page = 1
    if limit < 1:
        limit = 10

    # Get all projects
    projects = await get_project_directories()

    # Filter by status if provided (though all are active)
    if status:
        projects = [p for p in projects if p["status"] == status]

    # Calculate pagination
    total = len(projects)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit

    # Get paginated data
    paginated_projects = projects[start_idx:end_idx]

    # Get descriptions from paginated data
    def get_project_description(project_name):
        description_path = os.path.join(WORK_DIR, project_name, "description.txt")
        try:
            with open(description_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            # 파일이 없으면 빈 description.txt 파일 생성
            with open(description_path, "w") as f:
                f.write(f"## {project_name}")
            return ""

    projects = [
        Project(
            id=p["name"],
            name=p["name"],
            description=get_project_description(p["name"]),
            created_at=p["created_datetime"],
            status=p["status"],
            metadata={},
        )
        for p in paginated_projects
    ]

    return jsonify(
        {
            "total": total,
            "data": list(map(lambda p: p.model_dump(), projects)),
        }
    ), 200


@app.route("/projects/<string:project_id>/trials", methods=["GET"])
@project_exists(WORK_DIR)
async def get_trial_lists(project_id: str):
    project_db = SQLiteProjectDB(project_id)

    page = request.args.get("page", 1, type=int)
    limit = request.args.get("limit", 10, type=int)
    offset = (page - 1) * limit

    trials = project_db.get_trials_by_project(project_id, limit=limit, offset=offset)
    total_trials = len(project_db.get_all_trial_ids(project_id))

    return jsonify(
        {"total": total_trials, "data": [trial.model_dump() for trial in trials]}
    )


class FileNode(BaseModel):
    name: str
    type: str  # 'directory' or 'file'
    children: Optional[List["FileNode"]] = None


FileNode.model_rebuild()


async def scan_directory(path: str) -> FileNode:
    """비동기적으로 디렉토리를 스캔하여 파일 트리 구조를 생성합니다."""
    basename = os.path.basename(path)

    # 파일인지 확인
    if os.path.isfile(path):
        return FileNode(name=basename, type="file")

    # 디렉토리인 경우
    children = []
    try:
        # Convert scandir result to list for async iteration
        entries = await aiofiles.os.scandir(path)
        for item in entries:
            item_path = os.path.join(path, item.name)
            # 숨김 파일 제외
            if not item.name.startswith("."):
                children.append(await scan_directory(item_path))
    except PermissionError:
        pass

    return FileNode(
        name=basename,
        type="directory",
        children=sorted(children, key=lambda x: (x.type == "file", x.name)),
    )


@app.route("/projects/<string:project_id>/trials", methods=["POST"])
@project_exists(WORK_DIR)
async def create_trial(project_id: str):
    project_db = SQLiteProjectDB(project_id)

    data = await request.get_json()
    data["project_id"] = project_id
    trial = Trial(
        **data,
        created_at=datetime.now(tz=timezone.utc),
        status=Status.IN_PROGRESS,
        id=str(uuid.uuid4()),
    )

    project_db.set_trial(trial)
    return jsonify(trial.model_dump())


@app.route("/projects/<string:project_id>/trials/<string:trial_id>", methods=["GET"])
@project_exists(WORK_DIR)
async def get_trial(project_id: str, trial_id: str):
    project_db = SQLiteProjectDB(project_id)

    trial = project_db.get_trial(trial_id)
    if not trial:
        return jsonify({"error": "Trial not found"}), 404

    return jsonify(trial.model_dump())


@app.route("/projects/<string:project_id>/trials/<string:trial_id>", methods=["DELETE"])
@project_exists(WORK_DIR)
async def delete_trial(project_id: str, trial_id: str):
    project_db = SQLiteProjectDB(project_id)

    project_db.delete_trial(trial_id)
    return jsonify({"message": "Trial deleted successfully"})


@app.route("/projects/<string:project_id>/artifacts/files", methods=["GET"])
@project_exists(WORK_DIR)
async def get_artifact_file(project_id: str):
    """특정 파일의 내용을 비동기적으로 반환합니다."""
    file_path = request.args.get("path")
    if not file_path:
        return jsonify({"error": "File path is required"}), 400

    try:
        full_path = os.path.join(WORK_DIR, project_id, file_path)

        # 경로 검증 (디렉토리 트래버설 방지)
        if not os.path.normpath(full_path).startswith(
            os.path.normpath(os.path.join(WORK_DIR, project_id))
        ):
            return jsonify({"error": "Invalid file path"}), 403

        # 비동기로 파일 재 여부 확인
        if not await aiofiles.os.path.exists(full_path):
            return jsonify({"error": "File not found"}), 404

        if not await aiofiles.os.path.isfile(full_path):
            return jsonify({"error": "Path is not a file"}), 400

        # 파일 크기 체크
        stats = await aiofiles.os.stat(full_path)
        if stats.st_size > 10 * 1024 * 1024:  # 10MB 제한
            return jsonify({"error": "File too large"}), 400

        # 파일 확장자 체크
        _, ext = os.path.splitext(full_path)
        allowed_extensions = {".txt", ".yaml", ".yml", ".json", ".py", ".md"}
        if ext.lower() not in allowed_extensions:
            return jsonify({"error": "File type not supported"}), 400

        # 비동기로 파일 읽기
        async with aiofiles.open(full_path, "r", encoding="utf-8") as f:
            content = await f.read()

        return jsonify(
            {
                "content": content,
                "path": file_path,
                "size": stats.st_size,
                "last_modified": stats.st_mtime,
            }
        ), 200

    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 500


@app.route("/projects/<string:project_id>/upload", methods=["POST"])
@project_exists(WORK_DIR)
async def upload_files(project_id: str):
    # Setting upload
    raw_data_path = os.path.join(WORK_DIR, project_id, "raw_data")
    files = UploadSet()
    files.default_dest = raw_data_path
    configure_uploads(app, files)
    # List to hold paths of uploaded files
    uploaded_file_paths = []

    try:
        # Get all files from the request
        uploaded_files = (await request.files).getlist("files")

        if not uploaded_files:
            return jsonify({"error": "No files were uploaded"}), 400

        # Iterate over each file and save it
        for uploaded_file in uploaded_files:
            filename = await files.save(uploaded_file)
            uploaded_file_paths.append(os.path.join(raw_data_path, filename))

        return jsonify(
            {
                "message": "Files uploaded successfully",
                "filePaths": uploaded_file_paths,
            }
        ), 200

    except Exception as e:
        return jsonify(
            {"error": f"An error occurred while uploading files: {str(e)}"}
        ), 500


@app.route("/projects/<project_id>/trials/<trial_id>/parse", methods=["POST"])
async def parse_documents_endpoint(project_id, trial_id):
    task_id = ""
    try:
        # POST body에서 config 받기
        data = await request.get_json()
        if not data or "config" not in data:
            return jsonify({"error": "Config is required in request body"}), 400

        config = data["config"]

        # Trial 객체 가져오기
        project_db = SQLiteProjectDB(project_id)
        trial = project_db.get_trial(trial_id)
        if not trial:
            return jsonify({"error": f"Trial not found: {trial_id}"}), 404

        print(f"trial: {trial}")
        print(f"project_id: {project_id}")
        print(f"trial_id: {trial_id}")
        print(f"config: {config}")

        # Celery task 시작
        task = parse_documents.delay(
            project_id=project_id,
            trial_id=trial_id,
            config_str=yaml.dump(config),  # POST body의 config 사용
        )
        task_id = task.id
        print(f"task: {task}")

        # Trial 상태 업데이트
        trial.status = Status.IN_PROGRESS
        print(f"trial: {trial}")
        print(f"task: {task}")
        trial.parse_task_id = task.id
        project_db.set_trial(trial)

        return jsonify({"task_id": task.id, "status": "started"})
    except Exception as e:
        logger.error(f"Error starting parse task: {str(e)}", exc_info=True)
        return jsonify({"task_id": task_id, "status": "FAILURE", "error": str(e)}), 500


@app.route("/projects/<string:project_id>/tasks/<string:task_id>", methods=["GET"])
async def get_task_status(project_id: str, task_id: str):
    print(f"project_id: {project_id}")
    if task_id == "undefined":
        return jsonify({"status": "FAILURE", "error": "Task ID is undefined"}), 200
    else:
        # celery 상태 확인
        task = AsyncResult(task_id)
        print(f"task: {task.status}")
        try:
            return jsonify(
                {
                    "status": task.status,
                    "error": str(task.result) if task.failed() else None,
                }
            ), 200
        except Exception as e:
            return jsonify({"status": "FAILURE", "error": str(e)}), 500


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/chunk", methods=["POST"]
)
@project_exists(WORK_DIR)
@trial_exists(WORK_DIR)
async def start_chunking(project_id: str, trial_id: str):
    try:
        # Get JSON data from request and validate with Pydantic
        data = await request.get_json()
        chunk_request = ChunkRequest(**data)
        config = chunk_request.config

        project_db = SQLiteProjectDB(project_id)
        trial = project_db.get_trial(trial_id)
        if not trial:
            return jsonify({"error": f"Trial not found: {trial_id}"}), 404

        # Celery task 시작
        task = chunk_documents.delay(
            project_id=project_id,
            trial_id=trial_id,
            config_str=yaml.dump(config),  # POST body의 config 사용
        )
        task_id = task.id
        print(f"task: {task}")

        # Trial 상태 업데이트
        trial.status = Status.IN_PROGRESS
        print(f"trial: {trial}")
        print(f"task: {task}")
        trial.parse_task_id = task.id
        project_db.set_trial(trial)

        return jsonify({"task_id": task.id, "status": "started"})
    except Exception as e:
        logger.error(f"Error starting parse task: {str(e)}", exc_info=True)
        return jsonify({"task_id": task_id, "status": "FAILURE", "error": str(e)}), 500


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/qa", methods=["POST"]
)
@project_exists(WORK_DIR)
@trial_exists(WORK_DIR)
async def create_qa(project_id: str, trial_id: str):
    try:
        # Get JSON data from request and validate with Pydantic
        data = await request.get_json()

        project_db = SQLiteProjectDB(project_id)
        trial = project_db.get_trial(trial_id)
        if not trial:
            return jsonify({"error": f"Trial not found: {trial_id}"}), 404

        # Celery task 시작
        task = generate_qa_documents.delay(
            project_id=project_id,
            trial_id=trial_id,
            data=data,  # POST body 전체를 전달
        )
        task_id = task.id
        print(f"task: {task}")

        # Trial 상태 업데이트
        trial.status = Status.IN_PROGRESS
        print(f"trial: {trial}")
        print(f"task: {task}")
        trial.qa_task_id = task.id  # parse_task_id 대신 qa_task_id 사용
        project_db.set_trial(trial)

        return jsonify({"task_id": task.id, "status": "started"})
    except Exception as e:
        logger.error(f"Error starting QA generation task: {str(e)}", exc_info=True)
        return jsonify({"task_id": task_id, "status": "FAILURE", "error": str(e)}), 500


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/config", methods=["GET"]
)
@project_exists(WORK_DIR)
@trial_exists(WORK_DIR)
async def get_trial_config(project_id: str, trial_id: str):
    project_db = SQLiteProjectDB(project_id)
    trial = project_db.get_trial(trial_id)
    if not trial:
        return jsonify({"error": "Trial not found"}), 404
    return jsonify(trial.config), 200


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/config", methods=["POST"]
)
@project_exists(WORK_DIR)
@trial_exists(WORK_DIR)
async def set_trial_config(project_id: str, trial_id: str):
    project_db = SQLiteProjectDB(project_id)
    trial = project_db.get_trial(trial_id)
    if not trial:
        return jsonify({"error": "Trial not found"}), 404

    data = await request.get_json()
    if data.get("config", None) is not None:
        project_db.set_trial_config(trial_id, TrialConfig(**data["config"]))

    return jsonify({"message": "Config updated successfully"}), 200


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/validate", methods=["POST"]
)
@project_exists(WORK_DIR)
async def start_validate(project_id: str, trial_id: str):
    trial_config_path = os.path.join(WORK_DIR, project_id, "trials.db")
    trial_config_db = SQLiteProjectDB(trial_config_path)
    trial = trial_config_db.get_trial(trial_id)
    task_id = str(uuid.uuid4())
    response = Task(
        id=task_id,
        project_id=project_id,
        trial_id=trial_id,
        name=f"{trial_id}/validation",
        config_yaml=trial.config,
        status=Status.IN_PROGRESS,
        type=TaskType.VALIDATE,
        created_at=datetime.now(tz=timezone.utc),
    )
    await create_task(
        task_id,
        response,
        TaskType.VALIDATE,
        trial.config.qa_path,
        trial.config.corpus_path,
        trial.config.config_path,
    )

    return jsonify(response), 200


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/evaluate", methods=["POST"]
)
@project_exists(WORK_DIR)
async def start_evaluate(project_id: str, trial_id: str):
    evaluate_history_path = os.path.join(WORK_DIR, project_id, "evaluate_history.csv")
    if not os.path.exists(evaluate_history_path):
        evaluate_history_df = pd.DataFrame(
            columns=["trial_id", "save_dir", "corpus_path", "qa_path", "config_path"]
        )  # save_dir is to autorag trial directory
        evaluate_history_df.to_csv(evaluate_history_path, index=False)
    else:
        evaluate_history_df = pd.read_csv(evaluate_history_path)

    trial_config_path = os.path.join(WORK_DIR, project_id, "trials.db")
    trial_config_db = SQLiteProjectDB(trial_config_path)
    previous_config = trial_config_db.get_trial(trial_id).config
    print("previous_config: ", previous_config)
    trial = trial_config_db.get_trial(trial_id)
    trials_dir = os.path.join(WORK_DIR, project_id, "trials")

    data = await request.get_json()
    skip_validation = data.get("skip_validation", False)
    full_ingest = data.get("full_ingest", True)

    new_trial_dir = get_new_trial_dir(evaluate_history_df, trial.config, trials_dir)
    if os.path.exists(new_trial_dir):
        return jsonify(
            {
                "trial_dir": new_trial_dir,
                "error": "Exact same evaluation already run. "
                "Skipping but return the directory where the evaluation result is saved.",
            }
        ), 409
    task_id = str(uuid.uuid4())

    new_row = pd.DataFrame(
        [
            {
                "task_id": task_id,
                "trial_id": trial_id,
                "save_dir": new_trial_dir,
                "corpus_path": previous_config.corpus_path,
                "qa_path": previous_config.qa_path,
                "config_path": previous_config.config_path,
                "created_at": datetime.now(tz=timezone.utc),
            }
        ]
    )
    evaluate_history_df = pd.concat([evaluate_history_df, new_row], ignore_index=True)
    evaluate_history_df.reset_index(drop=True, inplace=True)
    evaluate_history_df.to_csv(evaluate_history_path, index=False)

    with open(trial.config.config_path, "r") as f:
        config_yaml = yaml.safe_load(f)
    task = Task(
        id=task_id,
        project_id=project_id,
        trial_id=trial_id,
        name=f"{trial_id}/evaluation",
        config_yaml=config_yaml,
        status=Status.IN_PROGRESS,
        type=TaskType.EVALUATE,
        created_at=datetime.now(tz=timezone.utc),
        save_path=new_trial_dir,
    )
    await create_task(
        task_id,
        task,
        run_start_trial,
        trial.config.qa_path,
        trial.config.corpus_path,
        os.path.dirname(new_trial_dir),
        trial.config.config_path,
        skip_validation,
        full_ingest,
    )

    task.model_dump()
    return jsonify(task.model_dump()), 202


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/report/open",
    methods=["GET"],
)
async def open_dashboard(project_id: str, trial_id: str):
    """
    Get a preparation task or run status for chat open.

    Args:
            project_id (str): The project ID
            trial_id (str): The trial ID

    Returns:
            JSON response with task status or error message
    """
    try:
        # Get the trial and search for the corresponding save_path
        evaluate_history_path = os.path.join(
            WORK_DIR, project_id, "evaluate_history.csv"
        )
        if not os.path.exists(evaluate_history_path):
            return jsonify({"error": "You need to run evaluation first"}), 400

        evaluate_history_df = pd.read_csv(evaluate_history_path)
        trial_raw = evaluate_history_df[evaluate_history_df["trial_id"] == trial_id]
        if trial_raw.empty or len(trial_raw) < 1:
            return jsonify({"error": "Trial ID not found"}), 404
        if len(trial_raw) >= 2:
            return jsonify({"error": "Duplicated trial ID found"}), 400

        trial_dir = trial_raw.iloc[0]["save_dir"]
        if not os.path.exists(trial_dir):
            return jsonify({"error": "Trial directory not found"}), 404
        if not os.path.isdir(trial_dir):
            return jsonify({"error": "Trial directory is not a directory"}), 500

        task_id = str(uuid.uuid4())
        response = Task(
            id=task_id,
            project_id=project_id,
            trial_id=trial_id,
            status=Status.IN_PROGRESS,
            type=TaskType.REPORT,
            created_at=datetime.now(tz=timezone.utc),
        )
        await create_task(task_id, response, run_dashboard, trial_dir)

        trial_config_path = os.path.join(WORK_DIR, project_id, "trials.db")
        trial_config_db = SQLiteProjectDB(trial_config_path)
        trial = trial_config_db.get_trial(trial_id)
        new_trial = trial.model_copy(deep=True)
        new_trial.report_task_id = task_id
        trial_config_db.set_trial(new_trial)

        return jsonify(response.model_dump()), 202

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/report/close",
    methods=["GET"],
)
async def close_dashboard(project_id: str, trial_id: str):
    trial_config_path = os.path.join(WORK_DIR, project_id, "trials.db")
    trial_config_db = SQLiteProjectDB(trial_config_path)
    trial = trial_config_db.get_trial(trial_id)
    report_pid = tasks[trial.report_task_id]["report_pid"]
    os.killpg(os.getpgid(report_pid), signal.SIGTERM)

    new_trial = trial.model_copy(deep=True)

    original_task = tasks[trial.report_task_id]["task"]
    original_task.status = Status.TERMINATED
    new_trial.report_task_id = None
    trial_config_db.set_trial(new_trial)

    return jsonify(original_task.model_dump()), 200


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/chat/open", methods=["GET"]
)
async def open_chat_server(project_id: str, trial_id: str):
    try:
        # Get the trial and search for the corresponding save_path
        evaluate_history_path = os.path.join(
            WORK_DIR, project_id, "evaluate_history.csv"
        )
        if not os.path.exists(evaluate_history_path):
            return jsonify({"error": "You need to run evaluation first"}), 400

        evaluate_history_df = pd.read_csv(evaluate_history_path)
        trial_raw = evaluate_history_df[evaluate_history_df["trial_id"] == trial_id]
        if trial_raw.empty or len(trial_raw) < 1:
            return jsonify({"error": "Trial ID not found"}), 404
        if len(trial_raw) >= 2:
            return jsonify({"error": "Duplicated trial ID found"}), 400

        trial_dir = trial_raw.iloc[0]["save_dir"]
        if not os.path.exists(trial_dir):
            return jsonify({"error": "Trial directory not found"}), 404
        if not os.path.isdir(trial_dir):
            return jsonify({"error": "Trial directory is not a directory"}), 500

        task_id = str(uuid.uuid4())
        response = Task(
            id=task_id,
            project_id=project_id,
            trial_id=trial_id,
            status=Status.IN_PROGRESS,
            type=TaskType.CHAT,
            created_at=datetime.now(tz=timezone.utc),
        )
        await create_task(task_id, response, run_chat, trial_dir)

        trial_config_path = os.path.join(WORK_DIR, project_id, "trials.db")
        trial_config_db = SQLiteProjectDB(trial_config_path)
        trial = trial_config_db.get_trial(trial_id)
        new_trial = trial.model_copy(deep=True)
        new_trial.chat_task_id = task_id
        trial_config_db.set_trial(new_trial)

        return jsonify(response.model_dump()), 202

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/projects/<string:project_id>/artifacts", methods=["GET"])
@project_exists(WORK_DIR)
async def get_project_artifacts(project_id: str):
    """프로젝트 아티팩트 디렉토리 구조를 비동기적으로 반환합니다."""
    try:
        project_path = os.path.join(WORK_DIR, project_id)

        # 특정 디렉토리만 스캔 (예: index 디렉토리)
        index_path = os.path.join(project_path, "raw_data")
        print(index_path)
        # 비동기로 디렉토리 존재 여부 확인
        if await aiofiles.os.path.exists(index_path):
            file_tree = await scan_directory(index_path)
            return jsonify(file_tree.model_dump()), 200
        else:
            return jsonify({"error": "Artifacts directory not found"}), 404
    except Exception as e:
        print(e)
        return jsonify({"error": f"Failed to scan artifacts: {str(e)}"}), 500


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/chat/close", methods=["GET"]
)
async def close_chat_server(project_id: str, trial_id: str):
    trial_config_path = os.path.join(WORK_DIR, project_id, "trials.db")
    trial_config_db = SQLiteProjectDB(trial_config_path)
    trial = trial_config_db.get_trial(trial_id)
    chat_pid = tasks[trial.chat_task_id]["chat_pid"]
    os.killpg(os.getpgid(chat_pid), signal.SIGTERM)

    new_trial = trial.model_copy(deep=True)

    original_task = tasks[trial.chat_task_id]["task"]
    original_task.status = Status.TERMINATED
    new_trial.chat_task_id = None
    trial_config_db.set_trial(new_trial)

    return jsonify(original_task.model_dump()), 200


@app.route("/projects/<string:project_id>/tasks", methods=["GET"])
@project_exists(WORK_DIR)
async def get_tasks(project_id: str):
    if not os.path.exists(os.path.join(WORK_DIR, project_id)):
        return jsonify({"error": f"Project name does not exist: {project_id}"}), 404

    evaluate_history_path = os.path.join(WORK_DIR, project_id, "evaluate_history.csv")
    if not os.path.exists(evaluate_history_path):
        evaluate_history_df = pd.DataFrame(
            columns=["trial_id", "save_dir", "corpus_path", "qa_path", "config_path"]
        )  # save_dir is to autorag trial directory
        evaluate_history_df.to_csv(evaluate_history_path, index=False)
    else:
        evaluate_history_df = pd.read_csv(evaluate_history_path)

    # Replace NaN values with None before converting to dict
    evaluate_history_df = evaluate_history_df.where(pd.notna(evaluate_history_df), -1)

    return jsonify(
        {
            "total": len(evaluate_history_df),
            "data": evaluate_history_df.to_dict(
                orient="records"
            ),  # Convert DataFrame to list of dictionaries
        }
    ), 200


@app.route("/projects/<string:project_id>/tasks/<string:task_id>", methods=["GET"])
@project_exists(WORK_DIR)
async def get_task(project_id: str, task_id: str):
    if not os.path.exists(os.path.join(WORK_DIR, project_id)):
        return jsonify({"error": f"Project name does not exist: {project_id}"}), 404
    task: Optional[Dict] = tasks.get(task_id, None)
    if task is None:
        return jsonify({"error": f"Task ID does not exist: {task_id}"}), 404
    response = task["task"]
    return jsonify(response.model_dump()), 200


@app.route("/env", methods=["POST"])
async def set_environment_variable():
    # Get JSON data from request
    data = await request.get_json()
    is_exist_env = load_dotenv(ENV_FILEPATH)
    if not is_exist_env:
        with open(ENV_FILEPATH, "w") as f:
            f.write("")

    try:
        # Validate request data using Pydantic model
        env_var = EnvVariableRequest(**data)

        if os.getenv(env_var.key, None) is None:
            # Set the environment variable
            os.environ[env_var.key] = env_var.value
            set_key(ENV_FILEPATH, env_var.key, env_var.value)
            return jsonify({}), 200
        else:
            os.environ[env_var.key] = env_var.value
            set_key(ENV_FILEPATH, env_var.key, env_var.value)
            return jsonify({}), 201

    except Exception as e:
        return jsonify(
            {
                "status": "error",
                "message": f"Failed to set environment variable: {str(e)}",
            }
        ), 400


@app.route("/env/<string:key>", methods=["GET"])
async def get_environment_variable(key: str):
    """
    Get environment variable by key.

    Args:
            key (str): The environment variable key to lookup

    Returns:
            Tuple containing response dictionary and status code
    """
    try:
        value = dotenv_values(ENV_FILEPATH).get(key, None)

        if value is None:
            return {"error": f"Environment variable '{key}' not found"}, 404

        return {"key": key, "value": value}, 200

    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}, 500


@app.route("/env", methods=["GET"])
async def get_all_env_keys():
    try:
        envs = dotenv_values(ENV_FILEPATH)
        return jsonify(dict(envs)), 200
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}, 500


@app.route("/env/<string:key>", methods=["DELETE"])
async def delete_environment_variable(key: str):
    try:
        value = dotenv_values(ENV_FILEPATH).get(key, None)
        if value is None:
            return {"error": f"Environment variable '{key}' not found"}, 404

        unset_key(ENV_FILEPATH, key)

        return {}, 200

    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}, 500


@click.command()
@click.option("--host", type=str, default="127.0.0.1", help="Host IP address")
@click.option("--port", type=int, default=8000, help="Port number")
def main(host: str = "127.0.0.1", port: int = 8000):
    uvicorn.run("app:app", host=host, port=port, reload=True, loop="asyncio")


if __name__ == "__main__":
    main()
