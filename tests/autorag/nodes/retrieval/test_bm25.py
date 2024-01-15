import os
import pathlib
import pickle

import pandas as pd

from autorag.nodes.retrieval import bm25

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
bm25_path = os.path.join(root_dir, "resources", "test_bm25_retrieval.pkl")
with open(bm25_path, 'rb') as r:
    bm25_corpus = pickle.load(r)

queries = [
    ["What is Visconde structure?", "What are Visconde structure?"],
    ["What is the structure of StrategyQA dataset in this paper?"],
    ["What's your favorite source of RAG framework?",
     "What is your source of RAG framework?",
     "Is RAG framework have source?"],
]

project_dir = os.path.join(root_dir, "resources", "sample_project")
qa_data = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))
corpus_data = pd.read_parquet(os.path.join(project_dir, "data", "corpus.parquet"))
previous_result = qa_data.sample(5)[["qid", "query"]]


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
