import asyncio
import os
import signal
import tempfile
import concurrent.futures
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import yaml
from pydantic import ValidationError
from quart import Quart, request, jsonify
from quart_cors import cors  # Import quart_cors to enable CORS
from quart_uploads import UploadSet, configure_uploads

from src.auth import require_auth
from src.evaluate_history import get_new_trial_dir
from src.run import (
    run_parser_start_parsing,
    run_chunker_start_chunking,
    run_qa_creation,
    run_start_trial,
    run_validate,
    run_dashboard,
    run_chat,
)
from src.schema import (
    ChunkRequest,
    ParseRequest,
    EnvVariableRequest,
    QACreationRequest,
    Project,
    Task,
    Status,
    TaskType,
    TrialCreateRequest,
    Trial,
    TrialConfig,
)

import nest_asyncio

from src.trial_config import PandasTrialDB
from src.validate import project_exists, trial_exists


nest_asyncio.apply()

app = Quart(__name__)
app = cors(
    app,
    allow_origin=["http://localhost:3000"],  # 구체적인 origin 지정
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
WORK_DIR = os.path.join(ROOT_DIR, "projects")


# Function to create a task
async def create_task(task_id: str, task: Task, func: Callable, *args) -> None:
    tasks[task_id] = {
        "function": func,
        "args": args,
        "error": None,
        "task": task,
    }
    await task_queue.put(task_id)


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

                # Run the function in a separate process
                future = loop.run_in_executor(
                    executor,
                    func,
                    *args,
                )
                task_futures[task_id] = future

                await future
                # Use future Results
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
                print(f"Task {task_id} failed with error: {e}")

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

    # Create a new project
    new_project_dir = os.path.join(WORK_DIR, data["name"])
    if not os.path.exists(new_project_dir):
        os.makedirs(new_project_dir)
        os.makedirs(os.path.join(new_project_dir, "parse"))
        os.makedirs(os.path.join(new_project_dir, "chunk"))
        os.makedirs(os.path.join(new_project_dir, "qa"))
        os.makedirs(os.path.join(new_project_dir, "project"))
        os.makedirs(os.path.join(new_project_dir, "config"))
        # Make trial_config.csv file
        _ = PandasTrialDB(os.path.join(new_project_dir, "trial_config.csv"))
    else:
        return jsonify({"error": f'Project name already exists: {data["name"]}'}), 400

    # save at 'description.txt' file
    with open(os.path.join(new_project_dir, "description.txt"), "w") as f:
        f.write(description)

    response = Project(
        id=data["name"],
        name=data["name"],
        description=description,
        created_at=datetime.now(),
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
                        item.stat().st_mtime
                    ),
                    "created_datetime": datetime.fromtimestamp(item.stat().st_ctime),
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
    trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
    trial_config_db = PandasTrialDB(trial_config_path)
    trial_ids = trial_config_db.get_all_config_ids()
    return jsonify(
        {
            "total": len(trial_ids),
            "data": list(
                map(lambda x: trial_config_db.get_trial(x).model_dump(), trial_ids)
            ),
        }
    )


@app.route("/projects/<string:project_id>/trials", methods=["POST"])
@project_exists(WORK_DIR)
async def create_new_trial(project_id: str):
    trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")

    data = await request.get_json()
    try:
        creation_request = TrialCreateRequest(**data)
    except ValidationError as e:
        return jsonify(
            {
                "error": f"Invalid request format : {e}",
            }
        ), 400

    trial_id = str(uuid.uuid4())

    request_dict = creation_request.model_dump()
    if request_dict["config"] is not None:
        config_path = os.path.join(
            WORK_DIR, project_id, "config", f"{str(uuid.uuid4())}.yaml"
        )
        with open(config_path, "w") as f:
            yaml.safe_dump(request_dict["config"], f)
    else:
        config_path = None
    request_dict["trial_id"] = trial_id
    request_dict["project_id"] = project_id
    request_dict["config_path"] = config_path
    request_dict["metadata"] = {}
    request_dict.pop("config")
    name = request_dict.pop("name")

    new_trial_config = TrialConfig(**request_dict)
    new_trial = Trial(
        id=trial_id,
        project_id=project_id,
        config=new_trial_config,
        name=name,
        status=Status.NOT_STARTED,
        created_at=datetime.now(),
    )
    trial_config_db = PandasTrialDB(trial_config_path)
    trial_config_db.set_trial(new_trial)
    return jsonify(new_trial.model_dump()), 202


@app.route("/projects/<string:project_id>/upload", methods=["POST"])
@project_exists(WORK_DIR)
async def upload_files(project_id: str):
    # Setting upload
    raw_data_path = os.path.join(WORK_DIR, project_id, "raw_data")
    files = UploadSet()
    files.default_dest = raw_data_path
    configure_uploads(app, files)
    try:
        filename = await files.save((await request.files)["file"])

        if not filename:
            return jsonify({"error": "No files were uploaded"}), 400

        return jsonify(
            {
                "message": "Files uploaded successfully",
                "filePaths": os.path.join(raw_data_path, filename),
            }
        ), 200

    except Exception as e:
        return jsonify(
            {"error": f"An error occurred while uploading files: {str(e)}"}
        ), 500


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/parse", methods=["POST"]
)
@project_exists(WORK_DIR)
@trial_exists(WORK_DIR)
async def start_parsing(project_id: str, trial_id: str):
    try:
        # Get JSON data from request and validate with Pydantic
        data = await request.get_json()
        parse_request = ParseRequest(**data)

        # Get the directory containing datasets
        dataset_dir = os.path.join(WORK_DIR, project_id, "parse", parse_request.name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        else:
            return jsonify(
                {"error": f"Parse dataset name already exists: {parse_request.name}"}
            ), 400

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as yaml_tempfile:
            with open(yaml_tempfile.name, "w") as w:
                yaml.safe_dump(parse_request.config, w)
            yaml_path = yaml_tempfile.name

        task_id = str(uuid.uuid4())
        response = Task(
            id=task_id,
            project_id=project_id,
            trial_id=trial_id,
            name=parse_request.name,
            config_yaml=parse_request.config,
            status=Status.IN_PROGRESS,
            type=TaskType.PARSE,
            created_at=datetime.now(),
            save_path=dataset_dir,
        )
        await create_task(
            task_id,
            response,
            run_parser_start_parsing,
            os.path.join(WORK_DIR, project_id, "raw_data", "*.pdf"),
            dataset_dir,
            yaml_path,
        )

        # Update to trial
        trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
        trial_config_db = PandasTrialDB(trial_config_path)
        previous_config = trial_config_db.get_trial_config(trial_id)
        new_config = previous_config.model_copy(deep=True)
        new_config.raw_path = os.path.join(
            dataset_dir, "0.parquet"
        )  # TODO: deal with multiple parse config later
        trial_config_db.set_trial_config(trial_id, new_config)

        return jsonify(response.model_dump()), 202

    except ValueError as ve:
        # Handle Pydantic validation errors
        return jsonify({"error": f"Validation error: {str(ve)}"}), 400

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


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

        # Get the directory containing datasets
        dataset_dir = os.path.join(WORK_DIR, project_id, "chunk", chunk_request.name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        else:
            return jsonify(
                {"error": f"Parse dataset name already exists: {chunk_request.name}"}
            ), 400

        trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
        trial_config_db = PandasTrialDB(trial_config_path)
        previous_config = trial_config_db.get_trial_config(trial_id)

        raw_filepath = previous_config.raw_path
        if raw_filepath is None or not raw_filepath or not os.path.exists(raw_filepath):
            return jsonify({"error": "Raw data path not found"}), 400

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as yaml_tempfile:
            with open(yaml_tempfile.name, "w") as w:
                yaml.safe_dump(chunk_request.config, w)
            yaml_path = yaml_tempfile.name

        task_id = str(uuid.uuid4())
        response = Task(
            id=task_id,
            project_id=project_id,
            trial_id=trial_id,
            name=chunk_request.name,
            config_yaml=chunk_request.config,
            status=Status.IN_PROGRESS,
            type=TaskType.CHUNK,
            created_at=datetime.now(),
            save_path=dataset_dir,
        )
        await create_task(
            task_id,
            response,
            run_chunker_start_chunking,
            raw_filepath,
            dataset_dir,
            yaml_path,
        )

        # Update to trial
        new_config: TrialConfig = previous_config.model_copy(deep=True)
        new_config.corpus_path = os.path.join(
            dataset_dir, "0.parquet"
        )  # TODO: deal with multiple chunk config later
        trial_config_db.set_trial_config(trial_id, new_config)

        return jsonify(response.model_dump()), 202

    except ValueError as ve:
        # Handle Pydantic validation errors
        return jsonify({"error": f"Validation error: {str(ve)}"}), 400

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/qa", methods=["POST"]
)
@project_exists(WORK_DIR)
@trial_exists(WORK_DIR)
async def create_qa(project_id: str, trial_id: str):
    data = await request.get_json()
    try:
        qa_creation_request = QACreationRequest(**data)
        dataset_dir = os.path.join(WORK_DIR, project_id, "qa")

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        save_path = os.path.join(dataset_dir, f"{qa_creation_request.name}.parquet")

        if os.path.exists(save_path):
            return jsonify(
                {"error": f"QA dataset name already exists: {qa_creation_request.name}"}
            ), 400

        trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
        trial_config_db = PandasTrialDB(trial_config_path)
        previous_config = trial_config_db.get_trial_config(trial_id)

        corpus_filepath = previous_config.corpus_path
        if (
            corpus_filepath is None
            or not corpus_filepath
            or not os.path.exists(corpus_filepath)
        ):
            return jsonify({"error": "Corpus data path not found"}), 400

        task_id = str(uuid.uuid4())
        response = Task(
            id=task_id,
            project_id=project_id,
            trial_id=trial_id,
            name=qa_creation_request.name,
            config_yaml={"preset": qa_creation_request.preset},
            status=Status.IN_PROGRESS,
            type=TaskType.QA,
            created_at=datetime.now(),
            save_path=save_path,
        )
        await create_task(
            task_id,
            response,
            run_qa_creation,
            qa_creation_request,
            corpus_filepath,
            dataset_dir,
        )

        # Update qa path
        new_config: TrialConfig = previous_config.model_copy(deep=True)
        new_config.qa_path = save_path
        trial_config_db.set_trial_config(trial_id, new_config)

        return jsonify(response.model_dump()), 202

    except Exception as e:
        return jsonify(
            {"status": "error", "message": f"Failed at creation of QA: {str(e)}"}
        ), 400


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/config", methods=["GET"]
)
@project_exists(WORK_DIR)
@trial_exists(WORK_DIR)
async def get_trial_config(project_id: str, trial_id: str):
    trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
    trial_config_db = PandasTrialDB(trial_config_path)
    trial_config = trial_config_db.get_trial_config(trial_id)
    return jsonify(trial_config.model_dump()), 200


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/config", methods=["POST"]
)
@project_exists(WORK_DIR)
@trial_exists(WORK_DIR)
async def set_trial_config(project_id: str, trial_id: str):
    trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
    trial_config_db = PandasTrialDB(trial_config_path)
    previous_config = trial_config_db.get_trial_config(trial_id)
    new_config = previous_config.model_copy(deep=True)
    data = await request.get_json()
    if data.get("raw_path", None) is not None:
        new_config.raw_path = data["raw_path"]
    if data.get("corpus_path", None) is not None:
        new_config.corpus_path = data["corpus_path"]
    if data.get("qa_path", None) is not None:
        new_config.qa_path = data["qa_path"]
    if data.get("config", None) is not None:
        new_config_path = os.path.join(
            WORK_DIR, project_id, "config", f"{str(uuid.uuid4())}.yaml"
        )
        with open(new_config_path, "w") as f:
            yaml.safe_dump(data["config"], f)
        new_config.config_path = new_config_path
    if data.get("metadata", None) is not None:
        new_config.metadata = data["metadata"]

    trial_config_db.set_trial_config(trial_id, new_config)
    return jsonify(new_config.model_dump()), 201


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/validate", methods=["POST"]
)
@project_exists(WORK_DIR)
async def start_validate(project_id: str, trial_id: str):
    trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
    trial_config_db = PandasTrialDB(trial_config_path)
    trial_config = trial_config_db.get_trial_config(trial_id)

    task_id = str(uuid.uuid4())
    with open(trial_config.config_path, "r") as f:
        config_yaml = yaml.safe_load(f)
    response = Task(
        id=task_id,
        project_id=project_id,
        trial_id=trial_id,
        name=f"{trial_id}/validation",
        config_yaml=config_yaml,
        status=Status.IN_PROGRESS,
        type=TaskType.VALIDATE,
        created_at=datetime.now(),
    )
    await create_task(
        task_id,
        response,
        run_validate,
        trial_config.qa_path,
        trial_config.corpus_path,
        trial_config.config_path,
    )

    return jsonify(response.model_dump()), 202


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

    trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
    trial_config_db = PandasTrialDB(trial_config_path)
    trial = trial_config_db.get_trial(trial_id)
    evaluate_dir = os.path.join(WORK_DIR, project_id, "project")

    # Update the trial progress to IN_PROGRESS
    updated_trial = trial.model_copy(deep=True)
    updated_trial.status = Status.IN_PROGRESS
    trial_config_db.set_trial(updated_trial)

    data = await request.get_json()
    skip_validation = data.get("skip_validation", False)
    full_ingest = data.get("full_ingest", True)

    new_trial_dir = get_new_trial_dir(evaluate_history_df, trial.config, evaluate_dir)
    if os.path.exists(new_trial_dir):
        return jsonify(
            {
                "trial_dir": new_trial_dir,
                "error": "Exact same evaluation already run. "
                "Skipping but return the directory where the evaluation result is saved.",
            }
        ), 409

    new_row = pd.DataFrame(
        [
            {
                "trial_id": trial_id,
                "save_dir": new_trial_dir,
                "corpus_path": trial.config.corpus_path,
                "qa_path": trial.config.qa_path,
                "config_path": trial.config.config_path,
            }
        ]
    )
    evaluate_history_df = pd.concat([evaluate_history_df, new_row], ignore_index=True)
    evaluate_history_df.reset_index(drop=True, inplace=True)
    evaluate_history_df.to_csv(evaluate_history_path, index=False)

    task_id = str(uuid.uuid4())
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
        created_at=datetime.now(),
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
        trial_id,
        trial_config_path,
    )

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
            created_at=datetime.now(),
        )
        await create_task(task_id, response, run_dashboard, trial_dir)

        trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
        trial_config_db = PandasTrialDB(trial_config_path)
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
    trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
    trial_config_db = PandasTrialDB(trial_config_path)
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
            created_at=datetime.now(),
        )
        await create_task(task_id, response, run_chat, trial_dir)

        trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
        trial_config_db = PandasTrialDB(trial_config_path)
        trial = trial_config_db.get_trial(trial_id)
        new_trial = trial.model_copy(deep=True)
        new_trial.chat_task_id = task_id
        trial_config_db.set_trial(new_trial)

        return jsonify(response.model_dump()), 202

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route(
    "/projects/<string:project_id>/trials/<string:trial_id>/chat/close", methods=["GET"]
)
async def close_chat_server(project_id: str, trial_id: str):
    trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
    trial_config_db = PandasTrialDB(trial_config_path)
    trial = trial_config_db.get_trial(trial_id)
    chat_pid = tasks[trial.chat_task_id]["chat_pid"]
    os.killpg(os.getpgid(chat_pid), signal.SIGTERM)

    new_trial = trial.model_copy(deep=True)

    original_task = tasks[trial.chat_task_id]["task"]
    original_task.status = Status.TERMINATED
    new_trial.chat_task_id = None
    trial_config_db.set_trial(new_trial)

    return jsonify(original_task.model_dump()), 200


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

    try:
        # Validate request data using Pydantic model
        env_var = EnvVariableRequest(**data)

        if os.getenv(env_var.key, None) is None:
            # Set the environment variable
            os.environ[env_var.key] = env_var.value
            return jsonify({}), 200
        else:
            os.environ[env_var.key] = env_var.value
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
        value = os.environ.get(key)

        if value is None:
            return {"error": f"Environment variable '{key}' not found"}, 404

        return {"key": key, "value": value}, 200

    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}, 500


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    app.run()
