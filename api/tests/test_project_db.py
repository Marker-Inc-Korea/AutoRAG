import os
import tempfile
import pytest
from datetime import datetime

from database.project_db import SQLiteProjectDB
from src.schema import Trial, TrialConfig


@pytest.fixture
def temp_project_dir():
    """임시 프로젝트 디렉토리 생성"""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['WORK_DIR'] = temp_dir
        yield temp_dir


@pytest.fixture
def project_db(temp_project_dir):
    """테스트용 프로젝트 DB 인스턴스 생성"""
    return SQLiteProjectDB("test_project")


@pytest.fixture
def sample_trial():
    """테스트용 Trial 객체 생성"""
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
        report_task_id="report_123",
        chat_task_id="chat_123"
    )


def test_db_initialization(temp_project_dir):
    """DB 초기화 테스트"""
    print("\n[테스트] DB 초기화")
    db = SQLiteProjectDB("test_project")
    db_path = os.path.join(temp_project_dir, "test_project", "project.db")
    print(f"- DB 파일 생성 확인: {db_path}")
    assert os.path.exists(db_path)


def test_set_and_get_trial(project_db, sample_trial):
    """Trial 저장 및 조회 테스트"""
    print("\n[테스트] Trial 저장 및 조회")
    print(f"- Trial 저장: ID={sample_trial.id}")
    project_db.set_trial(sample_trial)
    
    print("- Trial 조회 및 데이터 검증")
    retrieved_trial = project_db.get_trial(sample_trial.id)
    
    print("\n[Config 검증]")
    print(f"원본 Config: {sample_trial.config.model_dump()}")
    print(f"조회된 Config: {retrieved_trial.config.model_dump()}")
    
    assert retrieved_trial is not None, "Trial이 성공적으로 조회되어야 함"
    assert retrieved_trial.id == sample_trial.id, "Trial ID가 일치해야 함"
    assert retrieved_trial.config.model_dump() == sample_trial.config.model_dump(), "Config 데이터가 일치해야 함"
    print("- 검증 완료: 모든 필드가 일치함")


def test_get_nonexistent_trial(project_db):
    """존재하지 않는 Trial 조회 테스트"""
    print("\n[테스트] 존재하지 않는 Trial 조회")
    nonexistent_id = "nonexistent_id"
    print(f"- 존재하지 않는 ID로 조회: {nonexistent_id}")
    retrieved_trial = project_db.get_trial(nonexistent_id)
    assert retrieved_trial is None, "존재하지 않는 Trial은 None을 반환해야 함"
    print("- 검증 완료: None 반환 확인")


def test_set_trial_config(project_db, sample_trial):
    """Trial config 업데이트 테스트"""
    print("\n[테스트] Trial 설정 업데이트")
    print(f"- 기존 Trial 저장: ID={sample_trial.id}")
    project_db.set_trial(sample_trial)
    
    print("- 새로운 설정으로 업데이트")
    new_config = TrialConfig(
        trial_id="test_trial_1",
        project_id="test_project",
        raw_path="/new/path/to/raw",
        corpus_path="/new/path/to/corpus",
        qa_path="/new/path/to/qa",
        config_path="/new/path/to/config",
    )
    
    project_db.set_trial_config(sample_trial.id, new_config)
    retrieved_trial = project_db.get_trial(sample_trial.id)
    assert retrieved_trial.config.model_dump() == new_config.model_dump()
    print("- 검증 완료: 설정 업데이트 확인")


def test_get_trials_by_project(project_db, sample_trial):
    """프로젝트별 trial 목록 조회 테스트"""
    print("\n[테스트] 프로젝트별 Trial 목록 조회")
    print("- 첫 번째 Trial 저장")
    project_db.set_trial(sample_trial)
    
    print("- 두 번째 Trial 생성 및 저장")
    second_trial = Trial(
        id="test_trial_2",
        project_id="test_project",
        name="Test Trial 2",
        status="completed",
        created_at=datetime.now()
    )
    project_db.set_trial(second_trial)
    
    print("- 페이지네이션 테스트 (limit=1)")
    trials = project_db.get_trials_by_project("test_project", limit=1, offset=0)
    assert len(trials) == 1, "한 개의 Trial만 반환되어야 함"
    
    print("- 전체 Trial 조회 테스트")
    all_trials = project_db.get_trials_by_project("test_project", limit=10, offset=0)
    assert len(all_trials) == 2, "두 개의 Trial이 반환되어야 함"
    print(f"- 검증 완료: 총 {len(all_trials)}개의 Trial 확인")


def test_get_all_config_ids(project_db, sample_trial):
    """모든 config ID 조회 테스트"""
    project_db.set_trial(sample_trial)
    
    # config가 없는 trial 추가
    trial_without_config = Trial(
        id="test_trial_2",
        project_id="test_project",
        name="Test Trial 2",
        status="not_started",
        created_at=datetime.now()
    )
    project_db.set_trial(trial_without_config)
    
    config_ids = project_db.get_all_config_ids()
    assert len(config_ids) == 1
    assert config_ids[0] == sample_trial.id


def test_delete_trial(project_db, sample_trial):
    """Trial 삭제 테스트"""
    project_db.set_trial(sample_trial)
    project_db.delete_trial(sample_trial.id)
    assert project_db.get_trial(sample_trial.id) is None


def test_get_all_trial_ids(project_db, sample_trial):
    """모든 trial ID 조회 테스트"""
    project_db.set_trial(sample_trial)
    
    # 다른 프로젝트의 trial 추가
    other_trial = Trial(
        id="other_trial",
        project_id="other_project",
        name="Other Trial",
        status="not_started",
        created_at=datetime.now()
    )
    project_db.set_trial(other_trial)
    
    # 특정 프로젝트의 trial ID 조회
    project_trials = project_db.get_all_trial_ids(project_id="test_project")
    assert len(project_trials) == 1
    assert project_trials[0] == sample_trial.id
    
    # 모든 trial ID 조회
    all_trials = project_db.get_all_trial_ids()
    assert len(all_trials) == 2
    assert set(all_trials) == {sample_trial.id, other_trial.id}
