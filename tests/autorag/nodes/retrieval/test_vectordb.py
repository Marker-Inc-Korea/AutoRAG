import os
import pathlib

import pandas as pd

import chromadb
from llama_index.embeddings import OpenAIEmbedding

from autorag.nodes.retrieval import vectordb
from tests.autorag.nodes.retrieval.test_retrieval_base import (queries, project_dir, corpus_data, previous_result,
                                                               base_retrieval_test, base_retrieval_node_test)

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
chroma_path = os.path.join(root_dir, "resources", "test_vectordb_retrieval_chroma")

db = chromadb.PersistentClient(path=chroma_path)
collection = db.get_collection(name="test_vectordb_retrieval")

embedding_model = OpenAIEmbedding()

# 곧 없어질 녀석, 테스트용 코퍼스 만들기
node_chroma_path = os.path.join(root_dir, "resources", "sample_project", "resources", "chroma")
node_db = chromadb.PersistentClient(path=node_chroma_path)
node_collection = node_db.create_collection(name="openai", metadata={"hnsw:space": "cosine"})

node_test_corpus_path = os.path.join(root_dir, "resources", "sample_project", "data", "corpus.parquet")
corpus_df = pd.read_parquet(path=node_test_corpus_path, engine='pyarrow')

contents = corpus_df["contents"].tolist()
ids = corpus_df["doc_id"].tolist()

embedded_contents = embedding_model._get_text_embeddings(contents)
node_collection.add(ids=ids, embeddings=embedded_contents)


def test_vectordb_retrieval():
    top_k = 10
    original_vectordb = vectordb.__wrapped__
    id_result, score_result = original_vectordb(queries, top_k=top_k, collection=collection,
                                                embedding_model=embedding_model)
    base_retrieval_test(id_result, score_result, top_k)


def test_vectordb_node():
    result_df = vectordb(project_dir=project_dir, previous_result=previous_result, top_k=4, embedding_model="openai")
    base_retrieval_node_test(result_df)
