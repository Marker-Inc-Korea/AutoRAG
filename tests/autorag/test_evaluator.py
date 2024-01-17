import os.path
import pathlib
import shutil

import pandas as pd
import pytest

from autorag import Evaluator
from autorag.nodes.retrieval import bm25
from autorag.nodes.retrieval.run import run_retrieval_node
from autorag.schema import Node
from autorag.utils import validate_qa_dataset, validate_corpus_dataset

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
    assert node.modules[0].module == bm25


def test_start_trial(evaluator):
    evaluator.start_trial(os.path.join(resource_dir, 'simple.yaml'))
    assert os.path.exists(os.path.join(os.getcwd(), '0'))
    assert os.path.exists(os.path.join(os.getcwd(), 'data'))
    assert os.path.exists(os.path.join(os.getcwd(), 'resources'))
    assert os.path.exists(os.path.join(os.getcwd(), 'trial.json'))
    assert os.path.exists(os.path.join(os.getcwd(), '0', 'retrieve_node_line'))
    assert os.path.exists(os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'retrieval'))
    assert os.path.exists(os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'retrieval', 'bm25=>top_k_50.parquet'))
    expect_each_result_columns = ['retrieved_contents', 'retrieved_ids', 'retrieve_scores', 'retrieval_f1', 'retrieval_recall']
    each_result = pd.read_parquet(os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'retrieval', 'bm25=>top_k_50.parquet'))
    assert all([expect_column in each_result.columns for expect_column in expect_each_result_columns])
    expect_best_result_columns = ['qid', 'query', 'retrieval_gt', 'generation_gt',
                      'retrieved_contents', 'retrieved_ids', 'retrieve_scores', 'retrieval_f1', 'retrieval_recall']
    best_result = pd.read_parquet(os.path.join(os.getcwd(), '0', 'retrieve_node_line', 'retrieval', 'best.parquet'))
    assert all([expect_column in best_result.columns for expect_column in expect_best_result_columns])
