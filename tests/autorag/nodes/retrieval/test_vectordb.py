import os
import pathlib

import chromadb
from llama_index.embeddings import OpenAIEmbedding

from autorag.nodes.retrieval import vectordb
from tests.autorag.nodes.retrieval.test_retrieval_base import queries, project_dir, corpus_data, previous_result

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
chroma_path = os.path.join(root_dir, "resources", "test_vectordb_retrieval_chroma")

db = chromadb.PersistentClient(path=chroma_path)
collection = db.get_collection(name="test_vectordb_retrieval")

embedding_model = OpenAIEmbedding()


def test_vectordb_retrieval():
    top_k = 10
    original_vectordb = vectordb.__wrapped__
    id_result, score_result = original_vectordb(queries, top_k=top_k, collection=collection,
                                                embedding_model=embedding_model)
    assert len(id_result) == len(score_result) == 3
    for id_list, score_list in zip(id_result, score_result):
        assert isinstance(id_list, list)
        assert isinstance(score_list, list)
        assert len(id_list) == len(score_list) == top_k
        for _id, score in zip(id_list, score_list):
            assert isinstance(_id, str)
            assert isinstance(score, float)
        for i in range(1, len(score_list)):
            # Note: Lower distances indicate higher scores
            assert score_list[i - 1] <= score_list[i]


def test_vectordb_node():
    result_df = vectordb(project_dir=project_dir, previous_result=previous_result, top_k=4, embedding_model="openai")
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
                # Note: Lower distances indicate higher scores
                assert score_list[i - 1] <= score_list[i]

