import logging
import os
from typing import Optional, Dict

import pandas as pd

logger = logging.getLogger("AutoRAG")


def summary_df_to_yaml(summary_df: pd.DataFrame) -> Dict:
    """
    Convert trial summary dataframe to config yaml file.

    :param summary_df:
    :return: Dictionary of config yaml file.
        You can save this dictionary to yaml file.
    """
    # summary_df columns : 'node_line_name', 'node_type', 'best_module_filename',
    #                      'best_module_name', 'best_module_params', 'best_execution_time'
    grouped = summary_df.groupby('node_line_name')

    node_lines = [
        {
            'node_line_name': node_line_name,
            'nodes': [
                {
                    'node_type': row['node_type'],
                    'modules': [{
                        'module_type': row['best_module_name'],
                        **row['best_module_params']
                    }]
                }
                for _, row in node_line.iterrows()
            ]
        }
        for node_line_name, node_line in grouped
    ]
    return {'node_lines': node_lines}


def extract_pipeline(trial_path: str, output_path: Optional[str] = None) -> Dict:
    """
    Extract the optimal pipeline from evaluated trial.

    :param trial_path: The path to the trial directory that you want to extract the pipeline from.
        Must already be evaluated.
    :param output_path: Output path that pipeline yaml file will be saved.
        Must be .yaml or .yml file.
        If None, it does not save yaml file and just return dict values.
        Default is None.
    :return: The dictionary of the extracted pipeline.
    """
    summary_path = os.path.join(trial_path, 'summary.parquet')
    if not os.path.exists(summary_path):
        raise ValueError(f"summary.parquet does not exist in {trial_path}.")
    trial_summary_df = pd.read_parquet(summary_path)

    # config_path = os.path.join(trial_path, 'config.yaml')
    # if not os.path.exists(config_path):
    #     raise ValueError(f"config.yaml does not exist in {trial_path}.")
    # try:
    #     with open(config_path, 'r') as f:
    #         config = yaml.safe_load(f)
    # except yaml.YAMLError as exc:
    #     logger.error(exc)
    #     raise ValueError(f"config.yaml is invalid yaml file. {exc}")
    #
    # node_lines = config['node_lines']
