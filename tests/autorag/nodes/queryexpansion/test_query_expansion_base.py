import os
import pathlib

import shutil

import pytest
import pandas as pd

import chromadb
from llama_index.embeddings import OpenAIEmbedding

from autorag.nodes.retrieval.vectordb import vectordb_ingest


root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
project_dir = os.path.join(root_dir, "resources", "sample_project")
qa_data = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))
corpus_data = pd.read_parquet(os.path.join(project_dir, "data", "corpus.parquet"))
previous_result = qa_data.sample(5)

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
resource_path = os.path.join(root_dir, "resources")

embedding_model = OpenAIEmbedding()


@pytest.fixture
def ingested_vectordb_node():
    node_chroma_path = os.path.join(resource_path, "sample_project", "resources", "chroma")
    node_db = chromadb.PersistentClient(path=node_chroma_path)
    node_collection = node_db.create_collection(name="openai", metadata={"hnsw:space": "cosine"})

    node_test_corpus_path = os.path.join(resource_path, "sample_project", "data", "corpus.parquet")
    sample_project_corpus_df = pd.read_parquet(path=node_test_corpus_path)

    vectordb_ingest(node_collection, sample_project_corpus_df, embedding_model)

    assert node_collection.count() == 30
    yield node_collection
    if os.path.exists(node_chroma_path):
        shutil.rmtree(node_chroma_path)



def base_query_expansion_node_test(result_df):
    contents = result_df["retrieved_contents"].tolist()
    ids = result_df["retrieved_ids"].tolist()
    scores = result_df["retrieve_scores"].tolist()
    assert len(contents) == len(ids) == len(scores) == 5
    assert len(contents[0]) == len(ids[0]) == len(scores[0]) == 10
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