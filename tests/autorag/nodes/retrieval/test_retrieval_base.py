import os
import pathlib

from datetime import datetime

import pandas as pd

from autorag.nodes.retrieval.base import evenly_distribute_passages

queries = [
    ["What is Visconde structure?", "What are Visconde structure?"],
    ["What is the structure of StrategyQA dataset in this paper?"],
    ["What's your favorite source of RAG framework?",
     "What is your source of RAG framework?",
     "Is RAG framework have source?"],
]

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
project_dir = os.path.join(root_dir, "resources", "sample_project")
qa_data = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))
corpus_data = pd.read_parquet(os.path.join(project_dir, "data", "corpus.parquet"))
previous_result = qa_data.sample(5)

doc_id = ["doc1", "doc2", "doc3", "doc4", "doc5"]
contents = ["This is a test document 1.", "This is a test document 2.", "This is a test document 3.",
            "This is a test document 4.", "This is a test document 5."]
metadata = [{'datetime': datetime.now()} for _ in range(5)]
corpus_df = pd.DataFrame({"doc_id": doc_id, "contents": contents, "metadata": metadata})


def base_retrieval_test(id_result, score_result, top_k):
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


def base_retrieval_node_test(result_df):
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


def test_evenly_distribute_passages():
    ids = [[f'test-{i}-{j}' for i in range(10)] for j in range(3)]
    scores = [[i for i in range(10)] for _ in range(3)]
    top_k = 10
    new_ids, new_scores = evenly_distribute_passages(ids, scores, top_k)
    assert len(new_ids) == top_k
    assert len(new_scores) == top_k
    assert new_scores == [0, 1, 2, 3, 0, 1, 2, 0, 1, 2]
