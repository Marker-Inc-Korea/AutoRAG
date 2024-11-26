import os
import uuid

from celery import shared_task
from .base import TrialTask
from src.schema import (
    QACreationRequest,
    Status,
)
import logging
import yaml
from src.run import (
    run_parser_start_parsing,
    run_chunker_start_chunking,
    run_qa_creation,
)

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
ROOT_DIR = "/app"
ENV = os.getenv("AUTORAG_API_ENV", "dev")

# WORK_DIR 설정
if "AUTORAG_WORK_DIR" in os.environ:
    # 환경변수로 지정된 경우 해당 경로 사용
    WORK_DIR = os.getenv("AUTORAG_WORK_DIR")
else:
    # 환경변수가 없는 경우 기본값 사용
    WORK_DIR = os.path.join(ROOT_DIR, "projects")


@shared_task(bind=True, base=TrialTask)
def chunk_documents(self, project_id: str, trial_id: str, config_str: str):
    """
    Task for the chunk documents

    :param project_id: The project id of the trial
    :param trial_id: The id of the trial
    :param config_str: Configuration string for chunking
    :return: The result of the chunking (Maybe None?)
    """
    try:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status="chunking",
            progress=0,
            task_type="chunk",
        )

        # 청킹 작업 수행
        logger.info("Chunking documents")

        project_dir = os.path.join(WORK_DIR, project_id)
        config_dir = os.path.join(project_dir, "config")
        parsed_data_path = os.path.join(
            project_dir, "parse", f"parse_{trial_id}", "0.parquet"
        )
        chunked_data_dir = os.path.join(project_dir, "chunk", f"chunk_{trial_id}")
        os.makedirs(config_dir, exist_ok=True)
        config_dir = os.path.join(project_dir, "config")

        # config_str을 파이썬 딕셔너리로 변환 후 다시 YAML로 저장
        if isinstance(config_str, str):
            config_dict = yaml.safe_load(config_str)
        else:
            config_dict = config_str

        # YAML 파일 형식 확인
        if "modules" not in config_dict:
            config_dict = {"modules": config_dict}

        logger.debug(f"Chunking config_dict: {config_dict}")
        # YAML 파일 저장
        yaml_path = os.path.join(config_dir, f"chunk_config_{trial_id}.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True)

        result = run_chunker_start_chunking(
            parsed_data_path, chunked_data_dir, yaml_path
        )

        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status=Status.COMPLETED,
            progress=100,
            task_type="chunk",
        )
        return result
    except Exception as e:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status=Status.FAILED,
            progress=0,
            task_type="chunk",
            info={"error": str(e)},
        )
        raise


@shared_task(bind=True, base=TrialTask)
def generate_qa_documents(self, project_id: str, trial_id: str, data: dict):
    try:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status="generating_qa_docs",
            progress=0,
            task_type="qa_docs",
        )

        # QA 생성 작업 수행
        logger.info("Generating QA documents")

        project_dir = os.path.join(WORK_DIR, project_id)
        config_dir = os.path.join(project_dir, "config")
        corpus_filepath = os.path.join(
            project_dir, "chunk", f"chunk_{trial_id}", "0.parquet"
        )
        dataset_dir = os.path.join(project_dir, "qa", f"qa_{trial_id}")

        # 필요한 모든 디렉토리 생성
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)  # dataset_dir도 생성

        qa_creation_request = QACreationRequest(**data)
        result = run_qa_creation(qa_creation_request, corpus_filepath, dataset_dir)

        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status=Status.COMPLETED,
            progress=100,
            task_type="qa_docs",
        )
        return result
    except Exception as e:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status=Status.FAILED,
            progress=0,
            task_type="qa_docs",
            info={"error": str(e)},
        )
        raise


@shared_task(bind=True, base=TrialTask)
def parse_documents(self, project_id: str, config_str: str, glob_path: str = "*.*"):
    try:
        self.update_state_and_db(
            trial_id="",
            project_id=project_id,
            status=Status.IN_PROGRESS,
            progress=0,
            task_type="parse",
        )

        new_parse_id = str(uuid.uuid4())
        project_dir = os.path.join(WORK_DIR, project_id)
        raw_data_path = os.path.join(project_dir, "raw_data", glob_path)
        config_dir = os.path.join(project_dir, "config")
        parsed_data_path = os.path.join(project_dir, "parse", f"parse_{new_parse_id}")
        os.makedirs(config_dir, exist_ok=True)

        # config_str을 파이썬 딕셔너리로 변환 후 다시 YAML로 저장
        if isinstance(config_str, str):
            config_dict = yaml.safe_load(config_str)
        else:
            config_dict = config_str

        # YAML 파일 형식 확인
        if "modules" not in config_dict:
            config_dict = {"modules": config_dict}

        # YAML 파일 저장
        yaml_path = os.path.join(config_dir, f"parse_config_{new_parse_id}.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True)

        result = run_parser_start_parsing(raw_data_path, parsed_data_path, yaml_path)

        self.update_state_and_db(
            trial_id="",
            project_id=project_id,
            status=Status.COMPLETED,
            progress=100,
            task_type="parse",
        )
        return result
    except Exception as e:
        self.update_state_and_db(
            trial_id="",
            project_id=project_id,
            status=Status.FAILED,
            progress=0,
            task_type="parse",
            info={"error": str(e)},
        )
        raise
