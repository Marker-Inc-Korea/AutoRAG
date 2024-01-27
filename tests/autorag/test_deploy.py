import os
import pathlib
import shutil
import tempfile

import pandas as pd
import pytest
import yaml

from click.testing import CliRunner
from fastapi.testclient import TestClient
from autorag.deploy import summary_df_to_yaml, extract_best_config, Runner
from autorag.evaluator import Evaluator, cli

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
resource_dir = os.path.join(root_dir, 'resources')


@pytest.fixture
def evaluator():
    evaluator = Evaluator(os.path.join(resource_dir, 'qa_data_sample.parquet'),
                          os.path.join(resource_dir, 'corpus_data_sample.parquet'))
    yield evaluator
    paths_to_remove = ['0', 'data', 'resources', 'trial.json']

    for path in paths_to_remove:
        full_path = os.path.join(os.getcwd(), path)
        try:
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
        except FileNotFoundError:
            pass


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


def test_extract_best_config(pseudo_trial_path):
    yaml_dict = extract_best_config(pseudo_trial_path)
    assert yaml_dict == solution_dict
    with tempfile.NamedTemporaryFile(suffix='yaml', mode='w+t') as yaml_path:
        yaml_dict = extract_best_config(pseudo_trial_path, yaml_path.name)
        assert yaml_dict == solution_dict
        assert os.path.exists(yaml_path.name)
        yaml_dict = yaml.safe_load(yaml_path)
        assert yaml_dict == solution_dict


def test_runner(evaluator):
    evaluator.start_trial(os.path.join(resource_dir, 'simple.yaml'))

    def runner_test(runner: Runner):
        answer = runner.run('What is the best movie in Korea? Have Korea movie ever won Oscar?',
                            'retrieved_contents')
        assert len(answer) == 10
        assert isinstance(answer, list)
        assert isinstance(answer[0], str)

    runner = Runner.from_trial_folder(os.path.join(os.getcwd(), '0'))
    runner_test(runner)

    with tempfile.NamedTemporaryFile(suffix='yaml', mode='w+t') as yaml_path:
        extract_best_config(os.path.join(os.getcwd(), '0'), yaml_path.name)
        runner = Runner.from_yaml(yaml_path.name)
        runner_test(runner)


def test_runner_api_server(evaluator):
    import nest_asyncio
    nest_asyncio.apply()
    evaluator.start_trial(os.path.join(resource_dir, 'simple.yaml'))
    runner = Runner.from_trial_folder(os.path.join(os.getcwd(), '0'))

    client = TestClient(runner.app)

    # Use the TestClient to make a request to the server
    response = client.post('/run', json={
        'query': 'What is the best movie in Korea? Have Korea movie ever won Oscar?',
        'result_column': 'retrieved_contents'
    })
    assert response.status_code == 200
    assert 'retrieved_contents' in response.json()
    retrieved_contents = response.json()['retrieved_contents']
    assert len(retrieved_contents) == 10
    assert isinstance(retrieved_contents, list)
    assert isinstance(retrieved_contents[0], str)


def test_run_api():
    runner = CliRunner()
    result = runner.invoke(cli, ['run_api', '--config_path', 'test/path/test.yaml',
                                 '--host', '0.0.0.0', '--port', '8080'])
    assert result.exit_code == 1  # it will occur error because I run this test with a wrong yaml path.
    # But it means that the command is working well. If not, it will occur exit_code 2.
