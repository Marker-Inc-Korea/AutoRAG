import os
import pathlib
import shutil
import uuid
from datetime import datetime

import pytest
import yaml

from app import app, WORK_DIR
from src.schema import TrialConfig, Trial, Status
from src.trial_config import PandasTrialDB

tests_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = pathlib.PurePath(tests_dir).parent


@pytest.fixture
def client_for_test():
    yield app.test_client()


@pytest.fixture
def new_project_test_client():
    yield app.test_client()
    shutil.rmtree(os.path.join(WORK_DIR, "test_project"))


@pytest.fixture
def chunk_client():
    yield app.test_client()


@pytest.mark.asyncio
async def test_create_project_success(new_project_test_client):
    # Make request
    response = await new_project_test_client.post(
        "/projects",
        json={"name": "test_project", "description": "A test project"},
        headers={"Authorization": "Bearer good", "Content-Type": "application/json"},
    )

    # Assert response
    data = await response.get_json()
    assert response.status_code == 201
    assert data["name"] == "test_project"
    assert data["description"] == "A test project"
    assert data["status"] == "active"
    assert "created_at" in data
    assert data["id"] == "test_project"
    assert "metadata" in data

    assert os.path.exists(os.path.join(WORK_DIR, "test_project"))
    assert os.path.exists(os.path.join(WORK_DIR, "test_project", "parse"))
    assert os.path.exists(os.path.join(WORK_DIR, "test_project", "chunk"))
    assert os.path.exists(os.path.join(WORK_DIR, "test_project", "qa"))
    assert os.path.exists(os.path.join(WORK_DIR, "test_project", "project"))
    assert os.path.exists(os.path.join(WORK_DIR, "test_project", "config"))
    assert os.path.exists(os.path.join(WORK_DIR, "test_project", "trial_config.csv"))
    assert os.path.exists(os.path.join(WORK_DIR, "test_project", "description.txt"))

    with open(os.path.join(WORK_DIR, "test_project", "description.txt"), "r") as f:
        assert f.read() == "A test project"

    # Test GET of projects
    response = await new_project_test_client.get(
        "/projects?page=1&limit=10&status=active"
    )

    # Assert Response
    data = await response.get_json()
    assert response.status_code == 200
    assert data["total"] >= 1
    assert len(data["data"]) >= 1
    assert data["data"][0]["name"] == "test_project"
    assert data["data"][0]["status"] == "active"
    assert data["data"][0]["description"] == "A test project"

    with open(
        os.path.join(root_dir, "tests", "resources", "parsed_data", "baseball_1.pdf"),
        "rb",
    ) as f1, open(
        os.path.join(
            root_dir, "tests", "resources", "parsed_data", "korean_texts_two_page.pdf"
        ),
        "rb",
    ) as f2:
        # Prepare the data for upload
        data = [
            ("file", (os.path.basename(f1.name), f1)),
            ("file", (os.path.basename(f2.name), f2)),
        ]

    response = await new_project_test_client.post(
        "/projects/test_project/upload", files=data
    )

    assert response.status_code == 200
    data = await response.get_json()
    assert data["message"] == "Files uploaded successfully"
    assert "filePaths" in data
    assert len(data["filePaths"]) == 2
    assert os.path.dirname(data["filePaths"][0]).endswith("raw_data")
    assert os.path.exists(data["filePaths"][0])

    # duplicate project
    response = await new_project_test_client.post(
        "/projects",
        json={"name": "test_project", "description": "A test project one more time"},
        headers={"Authorization": "Bearer good", "Content-Type": "application/json"},
    )

    assert response.status_code == 400
    data = await response.get_json()
    assert "error" in data
    assert data["error"] == "Project name already exists: test_project"

    # Missing name at request
    response = await new_project_test_client.post(
        "/projects",
        json={"description": "A test project"},
        headers={"Authorization": "Bearer good", "Content-Type": "application/json"},
    )
    assert response.status_code == 400
    data = await response.get_json()
    assert "error" in data
    assert data["error"] == "Name is required"

    # Missing auth header
    response = await new_project_test_client.post(
        "/projects",
        json={"name": "test_project", "description": "A test project"},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 401

    # Invalid token
    response = await new_project_test_client.post(
        "/projects",
        json={"name": "test_project", "description": "A test project"},
        headers={"Authorization": "Bearer bad", "Content-Type": "application/json"},
    )
    assert response.status_code == 403


@pytest.fixture
def get_trial_list_client():
    yield app.test_client()
    shutil.rmtree(os.path.join(WORK_DIR, "test_project_get_trial_lists"))


@pytest.mark.asyncio
async def test_get_trial_lists(get_trial_list_client):
    project_id = "test_project_get_trial_lists"
    trial_id = str(uuid.uuid4())
    trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
    os.makedirs(os.path.join(WORK_DIR, project_id), exist_ok=True)

    # Create a trial config file
    trial_config = TrialConfig(
        trial_id=trial_id,
        project_id=project_id,
        raw_path="/path/to/raw",
        corpus_path="/path/to/corpus",
        qa_path="/path/to/qa",
        config_path="/path/to/config",
    )
    trial = Trial(
        id=trial_id,
        project_id=project_id,
        config=trial_config,
        name="Test Trial",
        status="not_started",
        created_at=datetime.now(),
    )
    trial_config_db = PandasTrialDB(trial_config_path)
    trial_config_db.set_trial(trial)

    response = await get_trial_list_client.get(f"/projects/{project_id}/trials")
    data = await response.get_json()

    assert response.status_code == 200
    assert data["total"] == 1
    assert data["data"][0]["id"] == trial_id
    assert data["data"][0]["config"]["project_id"] == project_id

    # Test @project_exists


@pytest.fixture
def create_new_trial_client():
    yield app.test_client()
    shutil.rmtree(os.path.join(WORK_DIR, "test_project_create_new_trial"))


@pytest.mark.asyncio
async def test_create_new_trial(create_new_trial_client):
    project_id = "test_project_create_new_trial"
    os.makedirs(os.path.join(WORK_DIR, project_id), exist_ok=True)
    trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")

    trial_create_request = {
        "name": "New Trial",
        "raw_path": None,
        "corpus_path": None,
        "qa_path": None,
        "config": None,
    }

    response = await create_new_trial_client.post(
        f"/projects/{project_id}/trials", json=trial_create_request
    )
    data = await response.get_json()

    assert response.status_code == 202
    assert data["name"] == "New Trial"
    assert data["project_id"] == project_id
    assert "id" in data

    # Verify the trial was added to the CSV
    trial_config_db = PandasTrialDB(trial_config_path)
    trial_ids = trial_config_db.get_all_config_ids()
    assert len(trial_ids) == 1
    assert trial_ids[0] == data["id"]


@pytest.fixture
def trial_config_client():
    client = app.test_client()
    yield client
    shutil.rmtree(os.path.join(WORK_DIR, "test_project"))


@pytest.mark.asyncio
async def test_get_trial_config(trial_config_client):
    project_id = "test_project"
    trial_id = "test_trial"
    response = await trial_config_client.post(
        "/projects",
        json={"name": project_id, "description": "A test project"},
        headers={"Authorization": "Bearer good", "Content-Type": "application/json"},
    )
    assert response.status_code == 201
    os.makedirs(os.path.join(WORK_DIR, project_id), exist_ok=True)
    trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
    trial_config_db = PandasTrialDB(trial_config_path)
    trial_config = TrialConfig(
        trial_id=trial_id,
        project_id=project_id,
        raw_path="raw_path",
        corpus_path="corpus_path",
        qa_path="qa_path",
        config_path="config_path",
        metadata={"key": "value"},
    )
    trial = Trial(
        id=trial_id,
        project_id=project_id,
        config=trial_config,
        name="Test Trial",
        status=Status.NOT_STARTED,
        created_at=datetime.now(),
    )
    trial_config_db.set_trial(trial)

    response = await trial_config_client.get(
        f"/projects/{project_id}/trials/{trial_id}/config"
    )
    data = await response.get_json()

    assert response.status_code == 200
    assert data["trial_id"] == trial_id
    assert data["project_id"] == project_id
    assert data["raw_path"] == "raw_path"
    assert data["corpus_path"] == "corpus_path"
    assert data["qa_path"] == "qa_path"
    assert data["config_path"] == "config_path"
    assert data["metadata"] == {"key": "value"}


@pytest.mark.asyncio
async def test_set_trial_config(trial_config_client):
    project_id = "test_project"
    trial_id = "test_trial"
    response = await trial_config_client.post(
        "/projects",
        json={"name": project_id, "description": "A test project"},
        headers={"Authorization": "Bearer good", "Content-Type": "application/json"},
    )
    assert response.status_code == 201

    os.makedirs(os.path.join(WORK_DIR, project_id), exist_ok=True)
    trial_config_path = os.path.join(WORK_DIR, project_id, "trial_config.csv")
    trial_config_db = PandasTrialDB(trial_config_path)
    trial_config = TrialConfig(
        trial_id=trial_id,
        project_id=project_id,
        raw_path="raw_path",
        corpus_path="corpus_path",
        qa_path="qa_path",
        config_path="config_path",
        metadata={"key": "value"},
    )
    trial = Trial(
        id=trial_id,
        project_id=project_id,
        config=trial_config,
        name="Test Trial",
        status=Status.NOT_STARTED,
        created_at=datetime.now(),
    )
    trial_config_db.set_trial(trial)

    new_config_data = {
        "raw_path": "new_raw_path",
        "corpus_path": "new_corpus_path",
        "qa_path": "new_qa_path",
        "config": {"jax": "children"},
        "metadata": {"new_key": "new_value"},
    }

    response = await trial_config_client.post(
        f"/projects/{project_id}/trials/{trial_id}/config", json=new_config_data
    )
    data = await response.get_json()

    assert response.status_code == 201
    assert data["trial_id"] == trial_id
    assert data["project_id"] == project_id
    assert data["raw_path"] == "new_raw_path"
    assert data["corpus_path"] == "new_corpus_path"
    assert data["qa_path"] == "new_qa_path"
    assert data["config_path"]
    with open(data["config_path"]) as f:
        assert yaml.safe_load(f) == {"jax": "children"}
    assert data["metadata"] == {"new_key": "new_value"}


@pytest.mark.asyncio
async def test_set_env_variable(client_for_test):
    os.environ.pop("test_key", None)
    response = await client_for_test.post(
        "/env",
        json={
            "key": "test_key",
            "value": "test_value",
        },
    )
    assert response.status_code == 200
    assert os.getenv("test_key") == "test_value"
    response = await client_for_test.post(
        "/env",
        json={
            "key": "test_key",
            "value": "test_value2",
        },
    )
    assert response.status_code == 201
    assert os.getenv("test_key") == "test_value2"


@pytest.mark.asyncio
async def test_get_env_variable(client_for_test):
    os.environ["test_key"] = "test_value"
    response = await client_for_test.get("/env/test_key")
    assert response.status_code == 200

    response = await client_for_test.get("/env/non_existent_key")
    assert response.status_code == 404
    data = await response.get_json()
    assert data["error"] == "Environment variable 'non_existent_key' not found"
