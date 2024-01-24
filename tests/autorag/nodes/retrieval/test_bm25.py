import os
import pathlib
import pickle
from datetime import datetime

import pandas as pd
import pytest

from autorag.nodes.retrieval import bm25
from autorag.nodes.retrieval.bm25 import bm25_ingest
from tests.autorag.nodes.retrieval.test_retrieval_base import queries, project_dir, corpus_data, previous_result

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
bm25_path = os.path.join(root_dir, "resources", "test_bm25_retrieval.pkl")
with open(bm25_path, 'rb') as r:
    bm25_corpus = pickle.load(r)


@pytest.fixture
def ingested_bm25_path():
    path = os.path.join(root_dir, "resources", "test_bm25_ingested.pkl")
    doc_id = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    contents = ["This is a test document 1.", "This is a test document 2.", "This is a test document 3.",
                "This is a test document 4.", "This is a test document 5."]
    metadata = [{'datetime': datetime.now()} for _ in range(5)]
    corpus_df = pd.DataFrame({"doc_id": doc_id, "contents": contents, "metadata": metadata})
    bm25_ingest(path, corpus_df)
    yield path
    os.remove(path)


def test_bm25_retrieval():
    top_k = 10
    original_bm25 = bm25.__wrapped__
    id_result, score_result = original_bm25(queries, top_k=top_k, bm25_corpus=bm25_corpus)
    assert len(id_result) == len(score_result) == 3
    for id_list, score_list in zip(id_result, score_result):
        assert isinstance(id_list, list)
        assert isinstance(score_list, list)
        assert len(id_list) == len(score_list) == top_k
        for _id, score in zip(id_list, score_list):
            assert isinstance(_id, str)
            assert isinstance(score, float)
        for i in range(1, len(score_list)):
            assert score_list[i - 1] >= score_list[i]


def test_bm25_node():
    result_df = bm25(project_dir=project_dir, previous_result=previous_result, top_k=4)
    contents = result_df["retrieved_contents"].tolist()
    ids = result_df["retrieved_ids"].tolist()
    scores = result_df["retrieve_scores"].tolist()
    assert len(contents) == len(ids) == len(scores) == 5
    assert len(contents[0]) == len(ids[0]) == len(scores[0]) == 4
    # id is matching with corpus.parquet
    for content_list, id_list, score_list in zip(contents, ids, scores):
        for i, (content, _id, score) in enumerate(zip(content_list, id_list, score_list)):
            assert isinstance(content, str)
            assert isinstance(_id, str)
            assert isinstance(score, float)
            assert _id in corpus_data["doc_id"].tolist()
            assert content == corpus_data[corpus_data["doc_id"] == _id]["contents"].values[0]
            if i >= 1:
                assert score_list[i - 1] >= score_list[i]


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
