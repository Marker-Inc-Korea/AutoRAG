import os
import pathlib
import tempfile

import chromadb
import pandas as pd
import pytest
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.nodes.retrieval.vectordb import vectordb_ingest

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
project_dir = os.path.join(root_dir, "resources", "sample_project")
qa_data = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))
corpus_data = pd.read_parquet(os.path.join(project_dir, "data", "corpus.parquet"))
previous_result = qa_data.sample(5)

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
resource_path = os.path.join(root_dir, "resources")

embedding_model = OpenAIEmbedding()


@pytest.fixture
def ingested_vectordb_node():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as node_chroma_path:
		node_db = chromadb.PersistentClient(path=node_chroma_path)
		node_collection = node_db.create_collection(
			name="openai", metadata={"hnsw:space": "cosine"}
		)

		node_test_corpus_path = os.path.join(
			resource_path, "sample_project", "data", "corpus.parquet"
		)
		sample_project_corpus_df = pd.read_parquet(path=node_test_corpus_path)

		vectordb_ingest(node_collection, sample_project_corpus_df, embedding_model)

		assert node_collection.count() == 30
		yield node_collection


def base_query_expansion_node_test(result_df):
	queries = result_df["queries"].tolist()
	assert len(queries) == 5
	assert all(isinstance(query, str) for query_list in queries for query in query_list)
