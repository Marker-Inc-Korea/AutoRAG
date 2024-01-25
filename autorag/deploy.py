import logging
import os
from typing import Optional, Dict

import pandas as pd
import yaml

from autorag.utils.util import load_summary_file

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
    trial_summary_df = load_summary_file(summary_path, dict_columns=['best_module_params'])
    yaml_dict = summary_df_to_yaml(trial_summary_df)
    if output_path is not None:
        with open(output_path, 'w') as f:
            yaml.dump(yaml_dict, f)
    return yaml_dict
