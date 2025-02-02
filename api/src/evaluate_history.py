import os
import shutil
import uuid
import json
from datetime import datetime
from typing import List

import pandas as pd

from src.schema import TrialConfig


def get_new_trial_dir(
    past_trial_configs: List[TrialConfig],
    new_trial_config: TrialConfig,
    project_dir: str,
):
    def trial_configs_to_dataframe(trial_configs: List[TrialConfig]) -> pd.DataFrame:
        # Convert list of TrialConfig to list of dictionaries
        trial_configs_dicts = [
            trial_config.model_dump() for trial_config in trial_configs
        ]
        # Create DataFrame from list of dictionaries
        df = pd.DataFrame(trial_configs_dicts)
        return df

    if len(past_trial_configs) == 0:
        new_dir_name = f"{new_trial_config.trial_id}-{str(uuid.uuid4())}"
        os.makedirs(os.path.join(project_dir, new_dir_name))
        return os.path.join(project_dir, new_dir_name, "0")  # New trial folder

    history_df = trial_configs_to_dataframe(past_trial_configs)
    duplicate_corpus_rows = history_df[
        history_df["corpus_name"] == new_trial_config.corpus_name
    ]
    print(f"Duplicate corpus rows: {duplicate_corpus_rows}")
    if len(duplicate_corpus_rows) == 0:  # If corpus data changed
        # Changed Corpus - ingest again (Make new directory - new save_dir)
        new_dir_name = f"{new_trial_config.trial_id}-{str(uuid.uuid4())}"
        os.makedirs(os.path.join(project_dir, new_dir_name))
        return os.path.join(project_dir, new_dir_name, "0")  # New trial folder
    duplicate_qa_rows = duplicate_corpus_rows[
        duplicate_corpus_rows["qa_name"] == new_trial_config.qa_name
    ]
    print(f"Duplicate qa rows: {duplicate_qa_rows}")
    if len(duplicate_qa_rows) == 0:  # If qa data changed
        # swap qa data from the existing project directory
        existing_project_dir = os.path.dirname(duplicate_qa_rows.iloc[0]["save_dir"])
        shutil.copy(
            new_trial_config.qa_path,
            os.path.join(existing_project_dir, "data", "qa.parquet"),
        )
    duplicate_config_rows = duplicate_qa_rows[
        duplicate_qa_rows["config"] == new_trial_config.config
    ]
    print(f"Duplicate config rows: {duplicate_config_rows}")
    if len(duplicate_config_rows) > 0:
        duplicate_row_save_paths = duplicate_config_rows["save_dir"].unique().tolist()
        print(f"Duplicate row save paths: {duplicate_row_save_paths}")
        return duplicate_row_save_paths[0]
    # Get the next trial folder
    existing_project_dir = os.path.dirname(duplicate_qa_rows.iloc[0]["save_dir"])
    latest_trial_name = get_latest_trial(
        os.path.join(existing_project_dir, "trial.json")
    )
    new_trial_name = str(int(latest_trial_name) + 1)
    return os.path.join(existing_project_dir, new_trial_name)


def get_latest_trial(file_path):
    try:
        # Load JSON file
        with open(file_path, "r") as f:
            trials = json.load(f)

        # Convert start_time to datetime objects and find the latest trial
        latest_trial = max(
            trials,
            key=lambda x: datetime.strptime(x["start_time"], "%Y-%m-%d %H:%M:%S"),
        )

        return latest_trial["trial_name"]

    except FileNotFoundError:
        print("Error: trial.json file not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
