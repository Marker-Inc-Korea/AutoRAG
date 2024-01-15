import os.path
import pathlib
import shutil

import pandas as pd
import pytest

from autorag.nodes.retrieval import bm25
from autorag.nodes.retrieval.run import run_retrieval_node

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
    modules = [bm25]
    module_params = [{'top_k': 4}]
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    qa_path = os.path.join(project_dir, "data", "qa.csv")
    retrieval_gt = pd.read_csv(qa_path)['retrieval_gt'].tolist()
    strategies = {
        'metrics': ['retrieval_f1', 'retrieval_recall'],
        'speed_threshold': 5,
    }
    previous_result = pd.read_csv(qa_path)
    best_result = run_retrieval_node(modules, module_params, previous_result, node_line_dir, retrieval_gt, strategies)
    assert os.path.exists(os.path.join(node_line_dir, "retrieval"))
    assert os.path.exists(os.path.join(node_line_dir, "retrieval", "bm25=>top_k_4.csv"))
    expect_columns = ['retrieved_contents', 'retrieved_ids', 'retrieve_scores', 'retrieval_f1', 'retrieval_recall']
    assert all([expect_column in best_result.columns for expect_column in expect_columns])
    # TODO: 채점이 이상함. 수정 필요 => csv를 parquet으로 수정
    # TODO: output column이 마음에 들지 않음. 수정 필요
