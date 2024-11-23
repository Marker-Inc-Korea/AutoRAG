import os
import tempfile
import pandas as pd
import pytest

from src.trial_config import SQLiteTrialDB
from src.schema import Trial, TrialConfig
from datetime import datetime


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        temp_path = tmp_file.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def sample_trial():
    return Trial(
        id="test_trial_1",
        project_id="test_project",
        config=TrialConfig(
            trial_id="test_trial_1",
            project_id="test_project",
            raw_path="/path/to/raw",
            corpus_path="/path/to/corpus",
            qa_path="/path/to/qa",
            config_path="/path/to/config",
        ),
        name="Test Trial",
        status="not_started",
        created_at=datetime.now(),
    )


def test_set_trial(temp_db_path, sample_trial):
    """
    Trial 저장 기본 기능 테스트
    - Trial 객체를 DB에 저장
    - 저장된 Trial의 기본 필드(id, project_id) 일치 여부 확인
    """
    trial_db = SQLiteTrialDB(temp_db_path)
    trial_db.set_trial(sample_trial)
    retrieved_trial = trial_db.get_trial(sample_trial.id)
    assert retrieved_trial is not None
    assert retrieved_trial.id == sample_trial.id
    assert retrieved_trial.project_id == sample_trial.project_id


def test_get_trial_existing(temp_db_path, sample_trial):
    """
    Trial 조회 상세 테스트 (존재하는 경우)
    - Trial 저장 후 조회
    - 모든 필드 일치 여부 확인
    - 특히 config 객체의 타입과 내용 정확성 검증
    """
    trial_db = SQLiteTrialDB(temp_db_path)
    trial_db.set_trial(sample_trial)
    retrieved_trial = trial_db.get_trial(sample_trial.id)
    assert retrieved_trial is not None
    assert retrieved_trial.id == sample_trial.id
    assert retrieved_trial.project_id == sample_trial.project_id
    assert isinstance(retrieved_trial.config, TrialConfig)
    assert retrieved_trial.config == sample_trial.config


def test_get_trial_nonexistent(temp_db_path):
    """
    Trial 조회 테스트 (존재하지 않는 경우)
    - 존재하지 않는 ID로 Trial 조회 시 None 반환 확인
    """
    trial_db = SQLiteTrialDB(temp_db_path)
    retrieved_trial = trial_db.get_trial("nonexistent_id")
    assert retrieved_trial is None


def test_set_trial_config(temp_db_path, sample_trial):
    """
    Trial Config 업데이트 테스트
    - 기존 Trial 저장
    - 새로운 Config로 업데이트
    - 업데이트된 Config 정확성 검증
    - Trial의 다른 필드는 유지되는지 확인
    """
    trial_db = SQLiteTrialDB(temp_db_path)
    trial_db.set_trial(sample_trial)
    new_config = TrialConfig(
        trial_id="test_trial_1",
        project_id="test_project",
        raw_path="/new/path/to/raw",
        corpus_path="/new/path/to/corpus",
        qa_path="/new/path/to/qa",
        config_path="/new/path/to/config",
    )
    trial_db.set_trial_config(sample_trial.id, new_config)
    retrieved_config = trial_db.get_trial(sample_trial.id).config
    assert retrieved_config is not None
    assert retrieved_config == new_config


def test_get_trial_config_existing(temp_db_path, sample_trial):
    """
    Trial Config 조회 테스트 (존재하는 경우)
    - Trial 저장 후 Config 조회
    - Config 객체의 세부 필드 일치 여부 확인
    """
    trial_db = SQLiteTrialDB(temp_db_path)
    trial_db.set_trial(sample_trial)
    retrieved_config = trial_db.get_trial(sample_trial.id).config
    assert retrieved_config is not None
    assert retrieved_config.trial_id == sample_trial.config.trial_id
    assert retrieved_config == sample_trial.config


def test_get_trial_config_nonexistent(temp_db_path):
    """
    Trial Config 조회 테스트 (존재하지 않는 경우)
    - 존재하지 않는 Trial의 Config 조회 시 None 반환 확인
    """
    trial_db = SQLiteTrialDB(temp_db_path)
    retrieved_trial = trial_db.get_trial('nonexistent_id')
    assert retrieved_trial is None


def test_get_all_config_ids(temp_db_path, sample_trial):
    """
    모든 Config ID 조회 테스트
    - Trial 저장 후 Config가 있는 Trial ID 목록 조회
    - 저장된 Trial ID가 목록에 포함되어 있는지 확인
    """
    trial_db = SQLiteTrialDB(temp_db_path)
    trial_db.set_trial(sample_trial)
    config_ids = trial_db.get_all_config_ids()
    assert len(config_ids) == 1
    assert config_ids[0] == sample_trial.id
