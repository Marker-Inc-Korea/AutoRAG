import os
import shutil

from celery import shared_task
from .base import TrialTask
from src.schema import (
    QACreationRequest,
    Status,
    TrialConfig,
)
import logging
import yaml
from src.run import (
    run_parser_start_parsing,
    run_chunker_start_chunking,
    run_qa_creation,
    run_start_trial,
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
def chunk_documents(
    self, project_id: str, config_str: str, parse_name: str, chunk_name: str
):
    """
    Task for the chunk documents

    :param project_id: The project id of the trial
    :param config_str: Configuration string for chunking
    :param parse_name: The name of the parsed data
    :param chunk_name: The name of the chunk
    """
    parsed_data_path = os.path.join(WORK_DIR, project_id, "parse", parse_name)
    if not os.path.exists(parsed_data_path):
        raise ValueError(f"parsed_data_path does not exist: {parsed_data_path}")

    try:
        self.update_state_and_db(
            trial_id="",
            project_id=project_id,
            status="chunking",
            progress=0,
            task_type="chunk",
        )

        # 청킹 작업 수행
        logger.info("Chunking documents")

        project_dir = os.path.join(WORK_DIR, project_id)
        config_dir = os.path.join(project_dir, "config")
        chunked_data_dir = os.path.join(project_dir, "chunk", chunk_name)
        os.makedirs(config_dir, exist_ok=True)
        config_dir = os.path.join(project_dir, "config")
        os.makedirs(chunked_data_dir, exist_ok=False)
    except Exception as e:
        self.update_state_and_db(
            trial_id="",
            project_id=project_id,
            status=Status.FAILED,
            progress=0,
            task_type="chunk",
            info={"error": str(e)},
        )
        raise

    try:
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
        yaml_path = os.path.join(config_dir, f"chunk_config_{chunk_name}.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True)

        result = run_chunker_start_chunking(
            parsed_data_path, chunked_data_dir, yaml_path
        )

        self.update_state_and_db(
            trial_id="",
            project_id=project_id,
            status=Status.COMPLETED,
            progress=100,
            task_type="chunk",
        )
        return result
    except Exception as e:
        self.update_state_and_db(
            trial_id="",
            project_id=project_id,
            status=Status.FAILED,
            progress=0,
            task_type="chunk",
            info={"error": str(e)},
        )
        if os.path.exists(chunked_data_dir):
            os.rmdir(chunked_data_dir)
        raise


@shared_task(bind=True, base=TrialTask)
def generate_qa_documents(self, project_id: str, request_data: QACreationRequest):
    try:
        self.update_state_and_db(
            trial_id="",
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
            project_dir, "chunk", request_data.chunked_name, "0.parquet"
        )
        if not os.path.exists(corpus_filepath):
            raise ValueError(f"corpus_filepath does not exist: {corpus_filepath}")

        dataset_dir = os.path.join(project_dir, "qa")

        os.makedirs(config_dir, exist_ok=True)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=False)

        run_qa_creation(request_data, corpus_filepath, dataset_dir)

        self.update_state_and_db(
            trial_id="",
            project_id=project_id,
            status=Status.COMPLETED,
            progress=100,
            task_type="qa_docs",
        )
    except Exception as e:
        self.update_state_and_db(
            trial_id="",
            project_id=project_id,
            status=Status.FAILED,
            progress=0,
            task_type="qa_docs",
            info={"error": str(e)},
        )
        raise


@shared_task(bind=True, base=TrialTask)
def parse_documents(
    self, project_id: str, config_str: str, parse_name: str, glob_path: str = "*.*"
):
    try:
        self.update_state_and_db(
            trial_id="",
            project_id=project_id,
            status=Status.IN_PROGRESS,
            progress=0,
            task_type="parse",
        )

        project_dir = os.path.join(WORK_DIR, project_id)
        raw_data_path = os.path.join(project_dir, "raw_data", glob_path)
        config_dir = os.path.join(project_dir, "config")
        parsed_data_path = os.path.join(project_dir, "parse", parse_name)
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(parsed_data_path, exist_ok=False)

    except Exception as e:
        self.update_state_and_db(
            trial_id="",
            project_id=project_id,
            status=Status.FAILED,
            progress=0,
            task_type="parse",
            info={"error": str(e)},
        )
    try:
        # config_str을 파이썬 딕셔너리로 변환 후 다시 YAML로 저장
        if isinstance(config_str, str):
            config_dict = yaml.safe_load(config_str)
        else:
            config_dict = config_str

        # YAML 파일 형식 확인
        if "modules" not in config_dict:
            config_dict = {"modules": config_dict}

        # YAML 파일 저장
        yaml_path = os.path.join(config_dir, f"parse_config_{parse_name}.yaml")
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
        if os.path.exists(parsed_data_path):
            shutil.rmtree(parsed_data_path)
        raise


@shared_task(bind=True, base=TrialTask)
def start_evaluate(
    self,
    project_id: str,
    trial_id: str,
    trial_config: TrialConfig,
    project_dir: str,
    skip_validation: bool = True,
    full_ingest: bool = True,
):
    try:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status=Status.IN_PROGRESS,
            progress=0,
            task_type="evaluate",
        )
        # Run the evaluation
        run_start_trial(
            qa_path=trial_config.qa_path,
            corpus_path=trial_config.corpus_path,
            project_dir=project_dir,
            yaml_path=trial_config.config_path,
            skip_validation=skip_validation,
            full_ingest=full_ingest,
        )
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status=Status.COMPLETED,
            progress=100,
            task_type="evaluate",
        )

    except Exception as e:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status=Status.FAILED,
            progress=0,
            task_type="evaluate",
            info={"error": str(e)},
        )
