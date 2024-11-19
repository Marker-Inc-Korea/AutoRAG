import os
import subprocess
from typing import Optional

import pandas as pd
from autorag import generator_models
from autorag.chunker import Chunker
from autorag.data.qa.schema import QA
from autorag.evaluator import Evaluator
from autorag.parser import Parser
from autorag.validator import Validator

from src.qa_create import default_create, fast_create, advanced_create
from src.schema import QACreationRequest, Status
from src.trial_config import PandasTrialDB


def run_parser_start_parsing(data_path_glob, project_dir, yaml_path):
    # Import Parser here if it's defined in another module
    parser = Parser(data_path_glob=data_path_glob, project_dir=project_dir)
    parser.start_parsing(yaml_path)


def run_chunker_start_chunking(raw_path, project_dir, yaml_path):
    # Import Parser here if it's defined in another module
    chunker = Chunker.from_parquet(raw_path, project_dir=project_dir)
    chunker.start_chunking(yaml_path)


def run_qa_creation(
    qa_creation_request: QACreationRequest, corpus_filepath: str, dataset_dir: str
):
    corpus_df = pd.read_parquet(corpus_filepath, engine="pyarrow")
    llm = generator_models[qa_creation_request.llm_config.llm_name](
        **qa_creation_request.llm_config.llm_params
    )

    if qa_creation_request.preset == "basic":
        qa: QA = default_create(
            corpus_df,
            llm,
            qa_creation_request.qa_num,
            qa_creation_request.lang,
            batch_size=8,
        )
    elif qa_creation_request.preset == "simple":
        qa: QA = fast_create(
            corpus_df,
            llm,
            qa_creation_request.qa_num,
            qa_creation_request.lang,
            batch_size=8,
        )
    elif qa_creation_request.preset == "advanced":
        qa: QA = advanced_create(
            corpus_df,
            llm,
            qa_creation_request.qa_num,
            qa_creation_request.lang,
            batch_size=8,
        )
    else:
        raise ValueError(f"Input not supported Preset {qa_creation_request.preset}")

    # dataset_dir will be folder ${PROJECT_DIR}/qa/
    qa.to_parquet(
        os.path.join(dataset_dir, f"{qa_creation_request.name}.parquet"),
        corpus_filepath,
    )


def run_start_trial(
    qa_path: str,
    corpus_path: str,
    project_dir: str,
    yaml_path: str,
    skip_validation: bool = True,
    full_ingest: bool = True,
    trial_id: Optional[str] = None,
    trial_config_path: Optional[str] = None,
):
    evaluator = Evaluator(qa_path, corpus_path, project_dir=project_dir)
    evaluator.start_trial(
        yaml_path, skip_validation=skip_validation, full_ingest=full_ingest
    )
    if trial_id is not None and trial_config_path is not None:
        # Update trial status
        trial_config_db = PandasTrialDB(trial_config_path)
        trial = trial_config_db.get_trial(trial_id)
        new_trial = trial.model_copy(deep=True)
        new_trial.status = Status.COMPLETED
        trial_config_db.set_trial(new_trial)


def run_validate(qa_path: str, corpus_path: str, yaml_path: str):
    validator = Validator(qa_path, corpus_path)
    validator.validate(yaml_path)


def run_dashboard(trial_dir: str):
    process = subprocess.Popen(
        ["autorag", "dashboard", "--trial_dir", trial_dir], start_new_session=True
    )
    return process.pid


def run_chat(trial_dir: str):
    process = subprocess.Popen(
        ["autorag", "run_web", "--trial_path", trial_dir], start_new_session=True
    )
    return process.pid
