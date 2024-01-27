import os.path
import pathlib
import shutil
import subprocess

import pandas as pd
import pytest

from autorag.evaluator import Evaluator
from autorag.nodes.retrieval import bm25, vectordb
from autorag.nodes.retrieval.run import run_retrieval_node
from autorag.schema import Node
from autorag.utils import validate_qa_dataset, validate_corpus_dataset
from autorag.utils.util import load_summary_file

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


def test_evaluator_init(evaluator):
    validate_qa_dataset(evaluator.qa_data)
    validate_corpus_dataset(evaluator.corpus_data)
    loaded_qa_data = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'qa.parquet'))
    loaded_corpus_data = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'corpus.parquet'))
    assert evaluator.qa_data.equals(loaded_qa_data)
    assert evaluator.corpus_data.equals(loaded_corpus_data)
    assert evaluator.project_dir == os.getcwd()


def test_load_node_line(evaluator):
    node_lines = Evaluator._load_node_lines(os.path.join(resource_dir, 'simple.yaml'))
    assert 'retrieve_node_line' in list(node_lines.keys())
    assert node_lines['retrieve_node_line'] is not None
    nodes = node_lines['retrieve_node_line']
    assert isinstance(nodes, list)
    assert len(nodes) == 1
    node = nodes[0]
    assert isinstance(node, Node)
    assert node.node_type == 'retrieval'
    assert node.run_node == run_retrieval_node
    assert node.strategy['metrics'] == ['retrieval_f1', 'retrieval_recall']
    assert node.modules[0].module_type == 'bm25'
    assert node.modules[1].module_type == 'vectordb'
    assert node.modules[0].module == bm25
    assert node.modules[1].module == vectordb


def test_start_trial(evaluator):
    evaluator.start_trial(os.path.join(resource_dir, 'simple.yaml'))
    assert os.path.exists(os.path.join(os.getcwd(), '0'))
    assert os.path.exists(os.path.join(os.getcwd(), 'data'))
    assert os.path.exists(os.path.join(os.getcwd(), 'resources'))
    assert os.path.exists(os.path.join(os.getcwd(), 'trial.json'))
    assert os.path.exists(os.path.join(os.getcwd(), '0', 'config.yaml'))
    assert os.path.exists(os.path.join(os.getcwd(), '0', 'retrieve_node_line'))
    assert os.path.exists(os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'retrieval'))
    assert os.path.exists(os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'retrieval', 'bm25=>top_k_10.parquet'))
    assert os.path.exists(os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'retrieval',
                                       'vectordb=>top_k_10-embedding_model_openai.parquet'))
    expect_each_result_columns = ['retrieved_contents', 'retrieved_ids', 'retrieve_scores', 'retrieval_f1',
                                  'retrieval_recall']
    each_result_bm25 = pd.read_parquet(
        os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'retrieval', 'bm25=>top_k_10.parquet'))
    each_result_vectordb = pd.read_parquet(
        os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'retrieval',
                     'vectordb=>top_k_10-embedding_model_openai.parquet'))
    assert all([expect_column in each_result_bm25.columns for expect_column in expect_each_result_columns])
    assert all([expect_column in each_result_vectordb.columns for expect_column in expect_each_result_columns])
    expect_best_result_columns = ['qid', 'query', 'retrieval_gt', 'generation_gt',
                                  'retrieved_contents', 'retrieved_ids', 'retrieve_scores', 'retrieval_f1',
                                  'retrieval_recall']
    best_result = pd.read_parquet(os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'retrieval',
                                               'best_bm25=>top_k_10.parquet'))
    assert all([expect_column in best_result.columns for expect_column in expect_best_result_columns])

    # test node line summary
    node_line_summary_path = os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'summary.parquet')
    assert os.path.exists(node_line_summary_path)
    node_line_summary_df = load_summary_file(node_line_summary_path,["best_module_params"])
    assert len(node_line_summary_df) == 1
    assert set(node_line_summary_df.columns) == {'node_type', 'best_module_filename',
                                                 'best_module_name', 'best_module_params', 'best_execution_time'}
    assert node_line_summary_df['node_type'][0] == 'retrieval'
    assert node_line_summary_df['best_module_filename'][0] == 'bm25=>top_k_10.parquet'
    assert node_line_summary_df['best_module_name'][0] == 'bm25'
    assert node_line_summary_df['best_module_params'][0] == {'top_k': 10}
    assert node_line_summary_df['best_execution_time'][0] > 0

    # test trial summary
    trial_summary_path = os.path.join(os.getcwd(), '0', 'summary.parquet')
    assert os.path.exists(trial_summary_path)
    trial_summary_df = load_summary_file(trial_summary_path, ["best_module_params"])
    assert len(trial_summary_df) == 1
    assert set(trial_summary_df.columns) == {'node_line_name', 'node_type', 'best_module_filename',
                                             'best_module_name', 'best_module_params', 'best_execution_time'}
    assert trial_summary_df['node_line_name'][0] == 'retrieve_node_line'
    assert trial_summary_df['node_type'][0] == 'retrieval'
    assert trial_summary_df['best_module_filename'][0] == 'bm25=>top_k_10.parquet'
    assert trial_summary_df['best_module_name'][0] == 'bm25'
    assert trial_summary_df['best_module_params'][0] == {'top_k': 10}
    assert trial_summary_df['best_execution_time'][0] > 0


def test_evaluator_cli(evaluator):
    result = subprocess.run(['autorag', 'evaluate', '--config', os.path.join(resource_dir, 'simple.yaml'),
                             '--qa_data_path', os.path.join(resource_dir, 'qa_data_sample.parquet'),
                             '--corpus_data_path', os.path.join(resource_dir, 'corpus_data_sample.parquet')])
    assert result.returncode == 0
    # check if the files are created
    assert os.path.exists(os.path.join(os.getcwd(), '0'))
    assert os.path.exists(os.path.join(os.getcwd(), 'data'))
    assert os.path.exists(os.path.join(os.getcwd(), 'resources'))
    assert os.path.exists(os.path.join(os.getcwd(), 'trial.json'))
    assert os.path.exists(os.path.join(os.getcwd(), '0', 'retrieve_node_line'))
    assert os.path.exists(os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'retrieval'))
    assert os.path.exists(os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'retrieval', 'bm25=>top_k_10.parquet'))
