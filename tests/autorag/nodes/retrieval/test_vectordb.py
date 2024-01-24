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


def test_vectordb_retrieval():
    top_k = 10
    original_vectordb = vectordb.__wrapped__
    id_result, score_result = original_vectordb(queries, top_k=top_k, collection=collection,
                                                embedding_model=embedding_model)
    base_retrieval_test(id_result, score_result, top_k)


def test_vectordb_node():
    result_df = vectordb(project_dir=project_dir, previous_result=previous_result, top_k=4, embedding_model="openai")
    base_retrieval_node_test(result_df)
