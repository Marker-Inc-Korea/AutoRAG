import os
import pathlib
import pickle

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
    path = os.path.join(root_dir, "resources", "test_bm25_ingested.pkl")
    bm25_ingest(path, corpus_df)
    yield path
    os.remove(path)


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
