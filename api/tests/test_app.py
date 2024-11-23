import pytest
import logging
import tempfile
import shutil
import os
from pathlib import Path
from datetime import datetime
import warnings
from pydantic import warnings as pydantic_warnings
import uuid

from app import app as quart_app, WORK_DIR

# Pydantic v2 deprecation 경고 무시
warnings.filterwarnings("ignore", category=pydantic_warnings.PydanticDeprecatedSince20)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TEST_HEADERS = {
    'Authorization': 'Bearer good'
}

# 전역 변수로 프로젝트 데이터 설정
project_data = {
    "name": f"test_project_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    "description": "Test Project",
    "metadata": {}
}
project_data_name = project_data["name"]
@pytest.fixture(scope="session")
def app_config():
    """Create test configuration for the entire session"""
    print("\n=== Setting up test session ===")
    
    # 테스트용 WORK_DIR 설정
    original_work_dir = os.environ.get('AUTORAG_WORK_DIR')
    test_work_dir = str(Path(__file__).resolve().parent.parent.parent / "test_projects")
    
    print(f"Original WORK_DIR: {original_work_dir}")
    print(f"Test WORK_DIR: {test_work_dir}")
    
    # 테스트 디렉토리 생성
    print(f"Creating test directory at: {test_work_dir}")
    os.makedirs(test_work_dir, exist_ok=True)
    
    # 환경변수 설정
    os.environ['AUTORAG_WORK_DIR'] = test_work_dir
    print(f"Set WORK_DIR environment variable to: {os.environ['AUTORAG_WORK_DIR']}")
    
    yield {
        'original_work_dir': original_work_dir,
        'test_work_dir': test_work_dir
    }
    
    print("\n=== Cleaning up test session ===")
    # if os.path.exists(test_work_dir):
    #     print(f"Removing test directory: {test_work_dir}")
    #     shutil.rmtree(test_work_dir)
    
    # 환경변수 복구
    if original_work_dir:
        os.environ['AUTORAG_WORK_DIR'] = original_work_dir
    else:
        os.environ.pop('AUTORAG_WORK_DIR', None)
    print("=== Session cleanup complete ===\n")

@pytest.fixture
async def app(app_config):
    """Create a test app instance"""
    app = quart_app
    
    # WORK_DIR 업데이트
    global AUTORAG_WORK_DIR
    AUTORAG_WORK_DIR = app_config['test_work_dir']
    app.config['AUTORAG_WORK_DIR'] = app_config['test_work_dir']
    app.config['ENV'] = 'test'
    
    return app

@pytest.fixture
async def test_client(app):
    """Create a test client"""
    logger.info("Creating test client")
    return app.test_client()

@pytest.mark.asyncio
async def test_create_project(test_client, app):
    """Test project creation endpoint"""
    logger.info(f"Testing project creation with data: {project_data}")
    logger.info(f"Using WORK_DIR: {app.config['AUTORAG_WORK_DIR']}")
    
    response = await test_client.post('/projects', 
                                    json=project_data,
                                    headers=TEST_HEADERS)
    
    status_code = response.status_code
    response_data = await response.get_json()
    logger.info(f"Response status: {status_code}")
    logger.info(f"Response data: {response_data}")
    
    assert status_code == 201
    
    # 프로젝트 디렉토리 경로 확인
    project_dir = Path(app.config['AUTORAG_WORK_DIR']) / project_data_name
    logger.info(f"Checking project directory at: {project_dir}")
    
    assert project_dir.exists(), f"Project directory does not exist: {project_dir}"
    for subdir in ['parse', 'chunk', 'qa', 'project', 'config', 'raw_data']:
        assert (project_dir / subdir).exists(), f"Subdirectory {subdir} does not exist"
    assert (project_dir / 'trials.db').exists(), "trials.db file does not exist"
    
    # description.txt 파일 검증
    desc_file = project_dir / 'description.txt'
    assert desc_file.exists(), "description.txt file does not exist"
    content = desc_file.read_text()
    assert content == project_data['description'], f"Expected description: {project_data['description']}, got: {content}"

@pytest.mark.asyncio
@pytest.mark.depends(on=['test_create_project'])  # 의존성 표시
async def test_create_trial_with_tasks(test_client, app):
    """Test trial creation with subsequent tasks"""
    # 기존 프로젝트 사용
    logger.info(f"Using existing project: {project_data_name}")
    
    # Trial 생성
    trial_data = {
        "name": "test_trial",
        "config": {
            "trial_id": "test_trial",
            "project_id": project_data["name"],
            "raw_path": "./raw_data/*.pdf"
        },
        "metadata": {}
    }
    
    logger.info(f"Creating trial with data: {trial_data}")
    trial_response = await test_client.post(
        f'/projects/{project_data_name}/trials',
        json=trial_data,
        headers=TEST_HEADERS
    )
    
    assert trial_response.status_code == 200
    trial_result = await trial_response.get_json()
    trial_id = trial_result['id']
    
    # Parse Task 생성
    parse_data = {
        "name": f"parse_{trial_id}",
        "path": "./raw_data/*.pdf",
        "config": {
            "modules": [{
                "module_type": "langchain_parse",
                "parse_method": ["pdfminer"]
            }]
        }
    }
    
    logger.info(f"Creating parse task with data: {parse_data}")
    parse_response = await test_client.post(
        f'/projects/{project_data["name"]}/trials/{trial_id}/parse',
        json=parse_data,
        headers=TEST_HEADERS
    )
    
    assert parse_response.status_code == 200
    parse_result = await parse_response.get_json()
    assert parse_result['status'] == 'in_progress'

@pytest.mark.asyncio
async def test_create_trial(test_client, app):
    """Test trial creation endpoint"""
    project_data_name = project_data["name"]
    logger.info(f"Creating test project with data: {project_data_name}")
    project_response = await test_client.post('/projects', 
                                            json=project_data, 
                                            headers=TEST_HEADERS)
    
    logger.info(f"Project response: {project_response}")
    assert project_response.status_code == 201
    
    # Trial 생성 - 간단한 데이터만 포함
    trial_data = {
        "name": "test_trial"
    }
    
    logger.info(f"Creating trial with data: {trial_data}")
    logger.info(f"Project path: {app.config['AUTORAG_WORK_DIR']}/{project_data_name}")
    
    response = await test_client.post(
        f'/projects/{project_data["name"]}/trials',
        json=trial_data,
        headers=TEST_HEADERS
    )
    
    status_code = response.status_code
    try:
        response_data = await response.get_json()
        logger.info(f"Response data: {response_data}")
    except Exception as e:
        logger.error(f"Failed to get response JSON: {e}")
        response_data = await response.get_data()
        logger.error(f"Raw response data: {response_data}")
    
    logger.info(f"Response status: {status_code}")
    
    assert status_code == 200, f"Expected 200, got {status_code}. Response: {response_data}"
    
    # Trial 데이터 검증
    assert response_data['name'] == trial_data['name']
    assert response_data['project_id'] == project_data_name
    assert response_data['status'] == 'in_progress'
    assert 'id' in response_data
    assert 'created_at' in response_data
    
    # Trial DB 파일 검증
    trial_db_path = Path(app.config['AUTORAG_WORK_DIR']) / project_data_name / 'trials.db'
    assert trial_db_path.exists(), f"Trial database does not exist at {trial_db_path}"

@pytest.mark.asyncio
async def test_create_trial_invalid_project(test_client):
    """Test trial creation with non-existent project"""
    trial_data = {
        "name": "test_trial",
        "config": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
    }
    
    logger.info("Testing trial creation with invalid project")
    response = await test_client.post(
        '/projects/nonexistent_project/trials',
        json=trial_data,
        headers=TEST_HEADERS
    )
    
    status_code = response.status_code
    response_data = await response.get_json()
    logger.info(f"Response status: {status_code}")
    logger.info(f"Response data: {response_data}")
    
    assert status_code == 404

@pytest.mark.asyncio
async def test_get_trial(test_client):
    """Test getting trial details"""
    project_data_name = project_data["name"]

    await test_client.post('/projects', json={"name": project_data_name}, headers=TEST_HEADERS)
    
    trial_response = await test_client.post(
        f'/projects/{project_data_name}/trials',
        json={"name": "test_trial"},
        headers=TEST_HEADERS
    )
    trial_data = await trial_response.get_json()
    trial_id = trial_data['id']
    
    response = await test_client.get(f'/projects/{project_data_name}/trials/{trial_id}', headers=TEST_HEADERS)
    assert response.status_code == 200
    
    data = await response.get_json()
    assert data['id'] == trial_id
    assert data['name'] == "test_trial"

@pytest.mark.asyncio
async def test_delete_trial(test_client):
    """Test trial deletion"""
    project_data_name = project_data_name
    await test_client.post('/projects', json={"name": project_data_name}, headers=TEST_HEADERS)
    trial_response = await test_client.post(
        f'/projects/{project_data_name}/trials',
        json={"name": "test_trial"},
        headers=TEST_HEADERS
    )
    trial_data = await trial_response.get_json()
    trial_id = trial_data['id']
    
    response = await test_client.delete(f'/projects/{project_data_name}/trials/{trial_id}', headers=TEST_HEADERS)
    assert response.status_code == 200
    
    get_response = await test_client.get(f'/projects/{project_data_name}/trials/{trial_id}', headers=TEST_HEADERS)
    assert get_response.status_code == 404

@pytest.mark.asyncio
async def test_environment_variables(test_client):
    """Test environment variable operations"""
    env_data = {"key": "TEST_KEY", "value": "test_value"}
    response = await test_client.post('/env', json=env_data, headers=TEST_HEADERS)
    assert response.status_code in [200, 201]
    
    response = await test_client.get(f'/env/{env_data["key"]}', headers=TEST_HEADERS)
    assert response.status_code == 200
    data = await response.get_json()
    assert data['value'] == env_data['value']
    
    response = await test_client.delete(f'/env/{env_data["key"]}', headers=TEST_HEADERS)
    assert response.status_code == 200
    
    response = await test_client.get(f'/env/{env_data["key"]}', headers=TEST_HEADERS)
    assert response.status_code == 404 