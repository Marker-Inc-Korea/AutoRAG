import os
import tempfile
import pandas as pd
import pytest

from src.trial_config import PandasTrialDB
from src.schema import Trial, TrialConfig
from datetime import datetime


@pytest.fixture
def temp_csv_path():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
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


def test_set_trial(temp_csv_path, sample_trial):
    trial_db = PandasTrialDB(temp_csv_path)
    trial_db.set_trial(sample_trial)
    df = pd.read_csv(temp_csv_path)
    assert len(df) == 1
    assert df.iloc[0]["id"] == sample_trial.id
    assert df.iloc[0]["project_id"] == sample_trial.project_id


def test_get_trial_existing(temp_csv_path, sample_trial):
    trial_db = PandasTrialDB(temp_csv_path)
    trial_db.set_trial(sample_trial)
    retrieved_trial = trial_db.get_trial(sample_trial.id)
    assert retrieved_trial is not None
    assert retrieved_trial.id == sample_trial.id
    assert retrieved_trial.project_id == sample_trial.project_id
    assert isinstance(retrieved_trial.config, TrialConfig)
    assert retrieved_trial.config == sample_trial.config

    retrieved_trial_config = trial_db.get_trial_config(sample_trial.id)
    assert retrieved_trial_config is not None
    assert isinstance(retrieved_trial_config, TrialConfig)
    assert retrieved_trial_config == sample_trial.config


def test_get_trial_nonexistent(temp_csv_path):
    trial_db = PandasTrialDB(temp_csv_path)
    retrieved_trial = trial_db.get_trial("nonexistent_id")
    assert retrieved_trial is None


def test_set_trial_config(temp_csv_path, sample_trial):
    trial_db = PandasTrialDB(temp_csv_path)
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
    retrieved_config = trial_db.get_trial_config(sample_trial.id)
    assert retrieved_config is not None
    assert retrieved_config == new_config

    retrieved_trial = trial_db.get_trial(sample_trial.id)
    assert retrieved_trial.id == sample_trial.id
    assert retrieved_trial.config == new_config


def test_get_trial_config_existing(temp_csv_path, sample_trial):
    trial_db = PandasTrialDB(temp_csv_path)
    trial_db.set_trial(sample_trial)
    retrieved_config = trial_db.get_trial_config(sample_trial.id)
    assert retrieved_config is not None
    assert retrieved_config.trial_id == sample_trial.config.trial_id
    assert retrieved_config == sample_trial.config


def test_get_trial_config_nonexistent(temp_csv_path):
    trial_db = PandasTrialDB(temp_csv_path)
    retrieved_config = trial_db.get_trial_config("nonexistent_id")
    assert retrieved_config is None


def test_get_all_config_ids(temp_csv_path, sample_trial):
    trial_db = PandasTrialDB(temp_csv_path)
    trial_db.set_trial(sample_trial)
    config_ids = trial_db.get_all_config_ids()
    assert len(config_ids) == 1
    assert config_ids[0] == sample_trial.id
