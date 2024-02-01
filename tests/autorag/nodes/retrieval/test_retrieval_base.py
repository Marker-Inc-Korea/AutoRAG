import os
import pathlib
import tempfile

from datetime import datetime

import pandas as pd
import pytest

from autorag.nodes.retrieval.base import evenly_distribute_passages, get_evaluation_result

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


@pytest.fixture
def pseudo_node_dir():
    summary_df = pd.DataFrame({
        'filename': ['bm25=>top_k_3.parquet', 'vectordb=>top_k_3-embedding_model_openai.parquet',
                      'vectordb=>top_k_3-embedding_model_huggingface.parquet'],
        'module_name': ['bm25', 'vectordb', 'vectordb'],
        'module_params': [
            {'top_k': 3},
            {'top_k': 3, 'embedding_model': 'openai'},
            {'top_k': 3, 'embedding_model': 'huggingface'},
        ],
        'execution_time': [1, 1, 1],
        'retrieval_f1': [0.1, 0.2, 0.3],
        'retrieval_recall': [0.2, 0.55, 0.5],
    })
    bm25_df = pd.DataFrame({
        'query': ['query-1', 'query-2', 'query-3'],
        'retrieved_ids': [['id-1', 'id-2', 'id-3'],
                          ['id-1', 'id-2', 'id-3'],
                          ['id-1', 'id-2', 'id-3']],
        'retrieve_scores': [[0.1, 0.2, 0.3],
                            [0.1, 0.2, 0.3],
                            [0.1, 0.2, 0.3]],
        'retrieval_f1': [0.05, 0.1, 0.15],
        'retrieval_recall': [0.1, 0.275, 0.25],
    })
    vector_openai_df = pd.DataFrame({
        'query': ['query-1', 'query-2', 'query-3'],
        'retrieved_ids': [['id-4', 'id-5', 'id-6'],
                          ['id-4', 'id-5', 'id-6'],
                          ['id-4', 'id-5', 'id-6']],
        'retrieve_scores': [[0.3, 0.4, 0.5],
                            [0.3, 0.4, 0.5],
                            [0.3, 0.4, 0.5]],
        'retrieval_f1': [0.15, 0.2, 0.25],
        'retrieval_recall': [0.3, 0.55, 0.5],
    })
    vector_huggingface_df = pd.DataFrame({
        'query': ['query-1', 'query-2', 'query-3'],
        'retrieved_ids': [['id-7', 'id-8', 'id-9'],
                          ['id-7', 'id-8', 'id-9'],
                          ['id-7', 'id-8', 'id-9']],
        'retrieve_scores': [[0.5, 0.6, 0.7],
                            [0.5, 0.6, 0.7],
                            [0.5, 0.6, 0.7]],
        'retrieval_f1': [0.25, 0.3, 0.35],
        'retrieval_recall': [0.5, 0.825, 0.75],
    })

    with tempfile.TemporaryDirectory() as node_dir:
        summary_df.to_parquet(os.path.join(node_dir, "summary.parquet"))
        bm25_df.to_parquet(os.path.join(node_dir, "bm25=>top_k_3.parquet"))
        vector_openai_df.to_parquet(os.path.join(node_dir, "vectordb=>top_k_3-embedding_model_openai.parquet"))
        vector_huggingface_df.to_parquet(
            os.path.join(node_dir, "vectordb=>top_k_3-embedding_model_huggingface.parquet"))
        yield node_dir


def test_get_evaluation_result(pseudo_node_dir):
    ids, scores = get_evaluation_result(pseudo_node_dir, ("bm25", "vectordb"))
    assert len(ids) == len(scores) == 2
    assert len(ids[0]) == len(scores[0]) == 3
    assert len(ids[1]) == len(scores[1]) == 3
    assert ids[0] == [['id-1', 'id-2', 'id-3'],
                      ['id-1', 'id-2', 'id-3'],
                      ['id-1', 'id-2', 'id-3']]
    assert scores[0] == [[0.1, 0.2, 0.3],
                         [0.1, 0.2, 0.3],
                         [0.1, 0.2, 0.3]]
    assert ids[1] == [['id-7', 'id-8', 'id-9'],
                      ['id-7', 'id-8', 'id-9'],
                      ['id-7', 'id-8', 'id-9']]
    assert scores[1] == [[0.5, 0.6, 0.7],
                         [0.5, 0.6, 0.7],
                         [0.5, 0.6, 0.7]]
