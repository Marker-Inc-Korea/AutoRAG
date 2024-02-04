import os.path
import pathlib
import shutil

import pandas as pd
import pytest
from llama_index.llms import MockLLM

from autorag import generator_models
from autorag.nodes.queryexpansion.run import evaluate_one_query_expansion_node
from autorag.nodes.queryexpansion import query_decompose, hyde
from autorag.nodes.queryexpansion.run import run_query_expansion_node
from autorag.nodes.retrieval import bm25
from autorag.utils.util import load_summary_file

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
resources_dir = os.path.join(root_dir, "resources")

sample_expanded_queries = [
    ["What is the purpose of rushing up the middle in football?",
     "Why are the first two plays often used for rushing up the middle in football?",
     "What are regular rush plays in football and how do they differ from rushing up the middle?"
     ],
    ["What are the prices of regular, mid, and premium gas?",
     " Why is there a 10 cent difference between the prices of regular, mid, and premium gas?",
     " Are there any specific factors or regulations that determine the pricing tiers of gas?"]
]
metrics = ['retrieval_f1', 'retrieval_recall']



@pytest.fixture
def node_line_dir():
    project_dir = os.path.join(resources_dir, "test_project")
    sample_project_dir = os.path.join(resources_dir, "sample_project")
    # copy & paste all folders and files in sample_project folder
    shutil.copytree(sample_project_dir, project_dir)

    test_trail_dir = os.path.join(project_dir, "test_trial")
    os.makedirs(test_trail_dir)
    node_line_dir = os.path.join(test_trail_dir, "test_node_line")
    os.makedirs(node_line_dir)
    yield node_line_dir
    # teardown
    shutil.rmtree(project_dir)


def test_evaluate_one_prompt_maker_node(node_line_dir):
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    qa_path = os.path.join(project_dir, "data", "qa.parquet")
    previous_result = pd.read_parquet(qa_path)
    sample_previous_result = previous_result.head(2)
    sample_retrieval_gt = sample_previous_result['retrieval_gt'].tolist()

    retrieval_funcs = [bm25, bm25]
    retrieval_params = [{'top_k': 1}, {'top_k': 2}]
    best_result = evaluate_one_query_expansion_node(retrieval_funcs, retrieval_params,
                                                    sample_expanded_queries, sample_retrieval_gt,
                                                    metrics, project_dir, sample_previous_result)
    assert isinstance(best_result, pd.DataFrame)
    assert all(metric_name in best_result.columns for metric_name in metrics)
    assert len(best_result) == len(sample_expanded_queries)


def base_query_expansion_test(best_result, node_line_dir):
    assert os.path.exists(os.path.join(node_line_dir, "query_expansion"))
    expect_columns = ['qid', 'query', 'generation_gt', 'retrieval_gt', 'queries',
                      'query_expansion_retrieval_f1', 'query_expansion_retrieval_recall']
    assert all([expect_column in best_result.columns for expect_column in expect_columns])
    assert os.path.exists(os.path.join(node_line_dir, "query_expansion", "0.parquet"))
    assert os.path.exists(os.path.join(node_line_dir, "query_expansion", "1.parquet"))
    # test summary feature
    summary_path = os.path.join(node_line_dir, "query_expansion", "summary.csv")
    assert os.path.exists(summary_path)
    summary_df = load_summary_file(summary_path)
    assert set(summary_df.columns) == {'filename', 'query_expansion_retrieval_f1', 'query_expansion_retrieval_recall',
                                       'module_name', 'module_params', 'execution_time', 'is_best'}
    assert len(summary_df) == 2
    assert summary_df['filename'][0] == "0.parquet"
    assert summary_df['module_name'][0] == "query_decompose"
    assert summary_df['module_params'][0] == {'llm': "openai", 'temperature': 0.2, 'batch': 7}
    assert summary_df['execution_time'][0] > 0
    assert summary_df['is_best'][0] == True  # is_best is np.bool_
    # test the best file is saved properly
    best_path = os.path.join(node_line_dir, "query_expansion", "best_0.parquet")
    assert os.path.exists(best_path)
    best_df = pd.read_parquet(best_path)
    assert all([expect_column in best_df.columns for expect_column in expect_columns])


def test_run_query_expansion_node(node_line_dir):
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    qa_path = os.path.join(project_dir, "data", "qa.parquet")
    previous_result = pd.read_parquet(qa_path)

    generator_models['mock'] = MockLLM
    modules = [query_decompose, hyde]
    module_params = [{'llm': "openai", 'temperature': 0.2, 'batch': 7}, {'llm': "mock"}]
    strategies = {
        'metrics': metrics,
        'speed_threshold': 5,
        'top_k': 4,
        'retrieval_modules': [{'module_type': 'bm25'}],
    }
    best_result = run_query_expansion_node(modules, module_params, previous_result, node_line_dir, strategies)
    base_query_expansion_test(best_result, node_line_dir)


def test_run_query_expansion_node_default(node_line_dir):
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    qa_path = os.path.join(project_dir, "data", "qa.parquet")
    previous_result = pd.read_parquet(qa_path)

    generator_models['mock'] = MockLLM
    modules = [query_decompose, hyde]
    module_params = [{'llm': "openai", 'temperature': 0.2, 'batch': 7}, {'llm': "mock"}]
    strategies = {
        'metrics': metrics
    }
    best_result = run_query_expansion_node(modules, module_params, previous_result, node_line_dir, strategies)
    base_query_expansion_test(best_result, node_line_dir)


def test_run_query_expansion_one_module(node_line_dir):
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    qa_path = os.path.join(project_dir, "data", "qa.parquet")
    previous_result = pd.read_parquet(qa_path)

    modules = [query_decompose]
    module_params = [{'llm': "openai", 'temperature': 0.2}]
    strategies = {
        'metrics': metrics
    }
    best_result = run_query_expansion_node(modules, module_params, previous_result, node_line_dir, strategies)
    assert set(best_result.columns) == {
        'qid', 'query', 'generation_gt', 'retrieval_gt', 'queries'  # automatically skip evaluation
    }
    summary_filepath = os.path.join(node_line_dir, "query_expansion", "summary.csv")
    assert os.path.exists(summary_filepath)
    summary_df = load_summary_file(summary_filepath)
    assert set(summary_df) == {
        'filename', 'module_name', 'module_params', 'execution_time', 'is_best'
    }
    best_filepath = os.path.join(node_line_dir, "query_expansion", f"best_{summary_df['filename'].values[0]}")
    assert os.path.exists(best_filepath)
