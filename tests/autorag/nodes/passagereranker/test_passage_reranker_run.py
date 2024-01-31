import os.path
import pathlib
import shutil
import tempfile

import pandas as pd
import pytest

from autorag.nodes.passagereranker import monot5
from autorag.nodes.retrieval import bm25
from autorag.nodes.retrieval.run import run_retrieval_node

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
resources_dir = os.path.join(root_dir, "resources")
qa_data = pd.DataFrame({
    'qid': ['id-1', 'id-2', 'id-3'],
    'query': ['query-1', 'query-2', 'query-3'],
    'retrieval_gt': [
        [['doc-1'], ['doc-2']],
        [['doc-3'], ['doc-4']],
        [['doc-5'], ['doc-6']],
    ],
    'generation_gt': [['generation-1'], ['generation-2'], ['generation-3']],
})
previous_result = pd.concat([qa_data, pd.DataFrame({
    'retrieved_ids': [
        ['doc-1', 'doc-3'],
        ['doc-2', 'doc-5'],
        ['doc-5', 'doc-6'],
    ],
    'retrieved_contents': [
        ['Enough for drinking water', 'Do you want to buy some?'],
        ['Just looking for a water bottle', 'I want to buy some water'],
        ['I want to buy some water', 'Who is son? He is great player in the world'],
    ],
    'retrieve_scores': [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
})], axis=1)

@pytest.fixture
def node_line_dir():
    with tempfile.TemporaryDirectory() as project_dir:
        data_dir = os.path.join(project_dir, "data")
        os.makedirs(data_dir)
        qa_data.to_parquet(os.path.join(data_dir, "qa.parquet"), index=False)
        trial_dir = os.path.join(project_dir, "trial_1")
        os.makedirs(trial_dir)
        node_line_dir = os.path.join(trial_dir, "node_line_1")
        os.makedirs(node_line_dir)
        yield node_line_dir

def test_run_passage_reranker_node(node_line_dir):
    modules = [monot5]
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    qa_path = os.path.join(project_dir, "data", "qa.parquet")
    strategies = {
        'metrics': ['retrieval_f1', 'retrieval_recall'],
        'speed_threshold': 5,
    }
