import os
import pathlib
import pickle
import tempfile
from datetime import datetime

import pandas as pd
import pytest

from autorag.nodes.retrieval import bm25
from autorag.nodes.retrieval.bm25 import bm25_ingest
from tests.autorag.nodes.retrieval.test_retrieval_base import (queries, project_dir, corpus_df, previous_result,
                                                               base_retrieval_test, base_retrieval_node_test)

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
bm25_path = os.path.join(root_dir, "resources", "test_bm25_retrieval.pkl")
with open(bm25_path, 'rb') as r:
    bm25_corpus = pickle.load(r)


@pytest.fixture
def ingested_bm25_path():
    with tempfile.NamedTemporaryFile(suffix='.pkl', mode='w+b') as path:
        bm25_ingest(path.name, corpus_df)
        yield path.name


def test_bm25_retrieval():
    top_k = 10
    original_bm25 = bm25.__wrapped__
    id_result, score_result = original_bm25(queries, top_k=top_k, bm25_corpus=bm25_corpus)
    base_retrieval_test(id_result, score_result, top_k)


def test_bm25_node():
    result_df = bm25(project_dir=project_dir, previous_result=previous_result, top_k=4)
    base_retrieval_node_test(result_df)


def test_bm25_ingest(ingested_bm25_path):
    with open(ingested_bm25_path, 'rb') as r:
        corpus = pickle.load(r)
    assert set(corpus.keys()) == {'tokens', 'passage_id'}
    assert isinstance(corpus['tokens'], list)
    assert isinstance(corpus['passage_id'], list)
    assert len(corpus['tokens']) == len(corpus['passage_id']) == 5
    assert set(corpus['passage_id']) == {'doc1', 'doc2', 'doc3', 'doc4', 'doc5'}

    bm25_origin = bm25.__wrapped__
    top_k = 2
    id_result, score_result = bm25_origin([['What is test document?'], ['What is test document number 2?']],
                                          top_k=top_k, bm25_corpus=corpus)
    assert len(id_result) == len(score_result) == 2
    for id_list, score_list in zip(id_result, score_result):
        assert isinstance(id_list, list)
        assert isinstance(score_list, list)
        for _id in id_list:
            assert isinstance(_id, str)
            assert _id in ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']


def test_duplicate_id_bm25_ingest(ingested_bm25_path):
    new_doc_id = ["doc4", "doc5", "doc6", "doc7", "doc8"]
    new_contents = ["This is a test document 4.", "This is a test document 5.", "This is a test document 6.",
                    "This is a test document 7.", "This is a test document 8."]
    new_metadata = [{'datetime': datetime.now()} for _ in range(5)]
    new_corpus_df = pd.DataFrame({"doc_id": new_doc_id, "contents": new_contents, "metadata": new_metadata})
    bm25_ingest(ingested_bm25_path, new_corpus_df)
    with open(ingested_bm25_path, 'rb') as r:
        corpus = pickle.load(r)
    assert len(corpus['tokens']) == len(corpus['passage_id']) == 8
