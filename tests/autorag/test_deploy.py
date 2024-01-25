import os
import pathlib
import tempfile

import pandas as pd
import pytest
import yaml

from autorag.deploy import summary_df_to_yaml, extract_pipeline

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
resource_dir = os.path.join(root_dir, 'resources')

summary_df = pd.DataFrame({
    'node_line_name': ['node_line_1', 'node_line_1', 'node_line_2'],
    'node_type': ['retrieval', 'rerank', 'generation'],
    'best_module_filename': ['bm25=>top_k_50.parquet', 'upr=>model_llama-2-havertz_chelsea.parquet',
                             'gpt-4=>top_p_0.9.parquet'],
    'best_module_name': ['bm25', 'upr', 'gpt-4'],
    'best_module_params': [{'top_k': 50}, {'model': 'llama-2', 'havertz': 'chelsea'}, {'top_p': 0.9}],
    'best_execution_time': [1.0, 0.5, 2.0]
})
solution_dict = {
    'node_lines': [
        {
            'node_line_name': 'node_line_1',
            'nodes': [
                {
                    'node_type': 'retrieval',
                    'modules': [
                        {
                            'module_type': 'bm25',
                            'top_k': 50
                        }
                    ]
                },
                {
                    'node_type': 'rerank',
                    'modules': [
                        {
                            'module_type': 'upr',
                            'model': 'llama-2',
                            'havertz': 'chelsea'
                        }
                    ]
                }
            ]
        },
        {
            'node_line_name': 'node_line_2',
            'nodes': [
                {
                    'node_type': 'generation',
                    'modules': [
                        {
                            'module_type': 'gpt-4',
                            'top_p': 0.9
                        }
                    ]
                }
            ]
        }
    ]
}


@pytest.fixture
def pseudo_trial_path():
    with tempfile.TemporaryDirectory() as project_dir:
        trial_path = os.path.join(project_dir, '0')
        os.makedirs(trial_path)
        summary_df.to_parquet(os.path.join(trial_path, 'summary.parquet'), index=False)
        yield trial_path


def test_summary_df_to_yaml():
    yaml_dict = summary_df_to_yaml(summary_df)
    assert yaml_dict == solution_dict


def test_extract_pipeline(pseudo_trial_path):
    yaml_dict = extract_pipeline(pseudo_trial_path)
    assert yaml_dict == solution_dict
    with tempfile.NamedTemporaryFile(suffix='yaml', mode='w+t') as yaml_path:
        yaml_dict = extract_pipeline(pseudo_trial_path, yaml_path.name)
        assert yaml_dict == solution_dict
        assert os.path.exists(yaml_path.name)
        yaml_dict = yaml.safe_load(yaml_path)
        assert yaml_dict == solution_dict
