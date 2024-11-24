import os
from celery import shared_task
from .base import TrialTask
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
from quart import jsonify, request
import logging
import yaml
import uuid
from datetime import datetime
import asyncio
from src.run import run_parser_start_parsing, run_chunker_start_chunking
    
# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
ROOT_DIR = "/app"
ENV = os.getenv('AUTORAG_API_ENV', 'dev')

# WORK_DIR 설정
if 'AUTORAG_WORK_DIR' in os.environ:
    # 환경변수로 지정된 경우 해당 경로 사용
    WORK_DIR = os.getenv('AUTORAG_WORK_DIR')
else:
    # 환경변수가 없는 경우 기본값 사용
    WORK_DIR = os.path.join(ROOT_DIR, "projects")

@shared_task(bind=True, base=TrialTask)
def chunk_documents(self, project_id: str, trial_id: str, config_str: str):
    """문서 청킹 작업"""
    try:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status="chunking",
            progress=0,
            task_type="chunk"
        )
        
        # 청킹 작업 수행
        logger.info("Chunking documents")
        
        project_dir = os.path.join(WORK_DIR, project_id)
        raw_data_path = os.path.join(project_dir, "raw_data", "*.pdf")
        config_dir = os.path.join(project_dir, "config")
        parsed_data_path = os.path.join(project_dir, "parse", f"parse_{trial_id}")
        chunked_data_path = os.path.join(project_dir, "chunk", f"chunk_{trial_id}")
        os.makedirs(config_dir, exist_ok=True)
        config_dir = os.path.join(project_dir, "config")

        # config_str을 파이썬 딕셔너리로 변환 후 다시 YAML로 저장
        if isinstance(config_str, str):
            config_dict = yaml.safe_load(config_str)
        else:
            config_dict = config_str

        # YAML 파일 형식 확인
        if 'modules' not in config_dict:
            config_dict = {'modules': config_dict}

        # YAML 파일 저장
        yaml_path = os.path.join(config_dir, f"parse_config_{trial_id}.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True)

        result = run_chunker_start_chunking(parsed_data_path, chunked_data_path, yaml_path)

        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status=Status.COMPLETED,
            progress=100,
            task_type="chunk"
        )
        return result
    except Exception as e:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status=Status.FAILED,
            progress=0,
            task_type="chunk",
            info={"error": str(e)}
        )
        raise

@shared_task(bind=True, base=TrialTask)
def parse_documents(self, project_id: str, trial_id: str, config_str: str):
    try:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status=Status.IN_PROGRESS,
            progress=0,
            task_type="parse"
        )

        project_dir = os.path.join(WORK_DIR, project_id)
        raw_data_path = os.path.join(project_dir, "raw_data", "*.pdf")
        config_dir = os.path.join(project_dir, "config")
        parsed_data_path = os.path.join(project_dir, "parse", f"parse_{trial_id}")
        os.makedirs(config_dir, exist_ok=True)

        # config_str을 파이썬 딕셔너리로 변환 후 다시 YAML로 저장
        if isinstance(config_str, str):
            config_dict = yaml.safe_load(config_str)
        else:
            config_dict = config_str

        # YAML 파일 형식 확인
        if 'modules' not in config_dict:
            config_dict = {'modules': config_dict}

        # YAML 파일 저장
        yaml_path = os.path.join(config_dir, f"parse_config_{trial_id}.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True)

        result = run_parser_start_parsing(raw_data_path, parsed_data_path, yaml_path)

        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status=Status.COMPLETED,
            progress=100,
            task_type="parse"
        )
        return result
    except Exception as e:
        self.update_state_and_db(
            trial_id=trial_id,
            project_id=project_id,
            status=Status.FAILED,
            progress=0,
            task_type="parse",
            info={"error": str(e)}
        )
        raise
