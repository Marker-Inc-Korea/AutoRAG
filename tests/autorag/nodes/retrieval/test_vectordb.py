import os
import pathlib
import shutil

import pytest
import pandas as pd

import chromadb
from llama_index.embeddings import OpenAIEmbedding

from autorag.nodes.retrieval import vectordb
from autorag.nodes.retrieval.vectordb import vectordb_ingest
from tests.autorag.nodes.retrieval.test_retrieval_base import (queries, project_dir, corpus_df, previous_result,
                                                               base_retrieval_test, base_retrieval_node_test)

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
resource_path = os.path.join(root_dir, "resources")
chroma_path = os.path.join(resource_path, "test_vectordb_retrieval_chroma")

db = chromadb.PersistentClient(path=chroma_path)
# collection = db.get_collection(name="test_vectordb_retrieval")

embedding_model = OpenAIEmbedding()


@pytest.fixture
def chroma_for_vectordb_node():
    node_chroma_path = os.path.join(resource_path, "sample_project", "resources", "chroma")
    node_db = chromadb.PersistentClient(path=node_chroma_path)
    node_collection = node_db.create_collection(name="openai", metadata={"hnsw:space": "cosine"})

    node_test_corpus_path = os.path.join(resource_path, "sample_project", "data", "corpus.parquet")
    sample_project_corpus_df = pd.read_parquet(path=node_test_corpus_path, engine='pyarrow')

    vectordb_ingest(node_collection, sample_project_corpus_df, embedding_model)

    yield node_collection
    if os.path.exists(node_chroma_path):
        shutil.rmtree(node_chroma_path)


def test_vectordb_retrieval():
    top_k = 10
    original_vectordb = vectordb.__wrapped__
    id_result, score_result = original_vectordb(queries, top_k=top_k, collection=collection,
                                                embedding_model=embedding_model)
    base_retrieval_test(id_result, score_result, top_k)


def test_vectordb_node(chroma_for_vectordb_node):
    result_df = vectordb(project_dir=project_dir, previous_result=previous_result, top_k=4, embedding_model="openai")
    base_retrieval_node_test(result_df)
