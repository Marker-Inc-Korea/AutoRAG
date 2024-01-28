import os.path
import pathlib
import shutil

import pandas as pd
import pytest

from autorag.nodes.queryexpansion import query_decompose
from autorag.nodes.queryexpansion.run import run_query_expansion_node


root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
resources_dir = os.path.join(root_dir, "resources")


@pytest.fixture
def node_line_dir():
    test_project_dir = os.path.join(resources_dir, "test_project")
    sample_project_dir = os.path.join(resources_dir, "sample_project")
    # copy & paste all folders and files in sample_project folder
    shutil.copytree(sample_project_dir, test_project_dir)

    test_trail_dir = os.path.join(test_project_dir, "test_trial")
    os.makedirs(test_trail_dir)
    node_line_dir = os.path.join(test_trail_dir, "test_node_line")
    os.makedirs(node_line_dir)
    yield node_line_dir
    # teardown
    shutil.rmtree(test_project_dir)


def test_run_retrieval_node(node_line_dir):
    modules = [query_decompose]
    module_params = [{'llm': "openai", 'temperature': 0.2}]
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    qa_path = os.path.join(project_dir, "data", "qa.parquet")
    strategies = {
        'metrics': ['retrieval_f1', 'retrieval_recall'],
        'speed_threshold': 5,
        'top_k': 4,
        'retrieval_module': [{'module_type': 'bm25'}],
    }
    previous_result = pd.read_parquet(qa_path)
    best_result = run_query_expansion_node(modules, module_params, previous_result, node_line_dir, strategies)
    assert os.path.exists(os.path.join(node_line_dir, "query_expansion"))
    expect_columns = ['qid', 'query', 'retrieval_gt', 'generation_gt',
                      'retrieved_contents', 'retrieved_ids', 'retrieve_scores', 'retrieval_f1', 'retrieval_recall']
    assert all([expect_column in best_result.columns for expect_column in expect_columns])
    # test summary feature
    summary_path = os.path.join(node_line_dir, "query_expansion", "summary.parquet")
    query_decompose_llm_temperature_path = os.path.join(node_line_dir, "query_expansion",
                                   "query_decompose=>llm_openai-temperature_0.2.parquet")
    assert os.path.exists(os.path.join(node_line_dir, "query_expansion",
                                       "query_decompose=>llm_openai-temperature_0.2.parquet"))
    query_decompose_llm_temperature_df = pd.read_parquet(query_decompose_llm_temperature_path)
    assert os.path.exists(summary_path)
    summary_df = pd.read_parquet(summary_path)
    assert set(summary_df.columns) == {'filename', 'retrieval_f1', 'retrieval_recall',
                                       'module_name', 'module_params', 'execution_time', 'is_best'}
    assert len(summary_df) == 1
    assert summary_df['filename'][0] == "query_decompose=>llm_openai-temperature_0.2.parquet"
    assert summary_df['retrieval_f1'][0] == query_decompose_llm_temperature_df['retrieval_f1'].mean()
    assert summary_df['retrieval_recall'][0] == query_decompose_llm_temperature_df['retrieval_recall'].mean()
    assert summary_df['module_name'][0] == "query_decompose"
    assert summary_df['module_params'][0] == {'llm': "openai", 'temperature': 0.2}
    assert summary_df['execution_time'][0] > 0
    assert summary_df['is_best'][0] == True # is_best is np.bool_
    # test the best file is saved properly
    best_path = os.path.join(node_line_dir, "query_expansion", "best_query_decompose=>llm_openai-temperature_0.2.parquet")
    assert os.path.exists(best_path)
    best_df = pd.read_parquet(best_path)
    assert all([expect_column in best_df.columns for expect_column in expect_columns])
