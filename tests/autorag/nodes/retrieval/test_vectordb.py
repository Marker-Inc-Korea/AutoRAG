import os
import pathlib
import shutil
import tempfile
import uuid
from datetime import datetime
from unittest.mock import patch

import chromadb
import pandas as pd
import pytest
from llama_index.core import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag import embedding_models
from autorag.nodes.retrieval import VectorDB
from autorag.nodes.retrieval.vectordb import vectordb_ingest, get_id_scores
from tests.autorag.nodes.retrieval.test_retrieval_base import (
	queries,
	corpus_df,
	previous_result,
	base_retrieval_test,
	base_retrieval_node_test,
	searchable_input_ids,
)
from tests.mock import mock_get_text_embedding_batch

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
resource_path = os.path.join(root_dir, "resources")

embedding_model = MockEmbedding(1536)


@pytest.fixture
def ingested_vectordb():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as chroma_path:
		db = chromadb.PersistentClient(path=chroma_path)
		collection = db.create_collection(
			name="test_vectordb_retrieval", metadata={"hnsw:space": "cosine"}
		)

		vectordb_ingest(collection, corpus_df, embedding_model)

		assert collection.count() == 5
		yield collection


@pytest.fixture
def empty_chromadb():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as chroma_path:
		db = chromadb.PersistentClient(path=chroma_path)
		collection = db.create_collection(
			name="test_vectordb_retrieval", metadata={"hnsw:space": "cosine"}
		)

		yield collection


@pytest.fixture
def project_dir_for_vectordb_node():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as test_project_dir:
		os.makedirs(os.path.join(test_project_dir, "resources"))
		chroma_path = os.path.join(test_project_dir, "resources", "chroma")
		os.makedirs(chroma_path)
		db = chromadb.PersistentClient(path=chroma_path)
		collection = db.create_collection(
			name="openai", metadata={"hnsw:space": "cosine"}
		)
		os.makedirs(os.path.join(test_project_dir, "data"))
		corpus_path = os.path.join(test_project_dir, "data", "corpus.parquet")
		corpus_df.to_parquet(corpus_path, index=False)
		vectordb_ingest(collection, corpus_df, embedding_model)

		yield test_project_dir


@pytest.fixture
def project_dir_for_vectordb_node_from_sample_project():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as test_project_dir:
		sample_project_dir = os.path.join(resource_path, "sample_project")
		# copy & paste all folders and files in the sample_project folder
		shutil.copytree(sample_project_dir, test_project_dir, dirs_exist_ok=True)

		chroma_path = os.path.join(test_project_dir, "resources", "chroma")
		os.makedirs(chroma_path)
		db = chromadb.PersistentClient(path=chroma_path)
		collection = db.create_collection(
			name="openai", metadata={"hnsw:space": "cosine"}
		)
		corpus_path = os.path.join(test_project_dir, "data", "corpus.parquet")
		local_corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")
		vectordb_ingest(collection, local_corpus_df, embedding_model)

		yield test_project_dir


@pytest.fixture
def vectordb_instance(project_dir_for_vectordb_node):
	vectordb = VectorDB(
		project_dir=project_dir_for_vectordb_node, embedding_model="openai"
	)
	yield vectordb


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_vectordb_retrieval(vectordb_instance):
	top_k = 4
	id_result, score_result = vectordb_instance._pure(
		queries,
		top_k=top_k,
	)
	base_retrieval_test(id_result, score_result, top_k)


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_vectordb_retrieval_ids(vectordb_instance):
	ids = [["doc2", "doc3"], ["doc1", "doc2"], ["doc4", "doc5"]]
	id_result, score_result = vectordb_instance._pure(
		queries,
		top_k=4,
		ids=ids,
	)
	assert id_result == ids
	assert len(id_result) == len(score_result) == 3
	assert all([len(score_list) == 2 for score_list in score_result])


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_vectordb_retrieval_ids_empty(vectordb_instance):
	ids = [["doc2", "doc3"], [], ["doc4"]]
	id_result, score_result = vectordb_instance._pure(
		queries,
		top_k=4,
		ids=ids,
	)
	assert id_result == ids
	assert len(id_result) == len(score_result) == 3
	assert len(score_result[0]) == 2
	assert len(score_result[1]) == 0
	assert len(score_result[2]) == 1


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_vectordb_node(project_dir_for_vectordb_node_from_sample_project):
	result_df = VectorDB.run_evaluator(
		project_dir=project_dir_for_vectordb_node_from_sample_project,
		previous_result=previous_result,
		top_k=4,
		embedding_model="openai",
	)
	base_retrieval_node_test(result_df)


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_vectordb_node_ids(project_dir_for_vectordb_node_from_sample_project):
	result_df = VectorDB.run_evaluator(
		project_dir=project_dir_for_vectordb_node_from_sample_project,
		previous_result=previous_result,
		top_k=4,
		embedding_model="openai",
		ids=searchable_input_ids,
	)
	contents = result_df["retrieved_contents"].tolist()
	ids = result_df["retrieved_ids"].tolist()
	scores = result_df["retrieve_scores"].tolist()
	assert len(contents) == len(ids) == len(scores) == 5
	assert len(contents[0]) == len(ids[0]) == len(scores[0]) == 2
	assert ids[0] == searchable_input_ids[0]


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_duplicate_id_vectordb_ingest(ingested_vectordb):
	vectordb_ingest(ingested_vectordb, corpus_df, embedding_model)
	assert ingested_vectordb.count() == 5

	new_doc_id = ["doc4", "doc5", "doc6", "doc7", "doc8"]
	new_contents = [
		"This is a test document 4.",
		"This is a test document 5.",
		"This is a test document 6.",
		"This is a test document 7.",
		"This is a test document 8.",
	]
	new_metadata = [{"datetime": datetime.now()} for _ in range(5)]
	new_corpus_df = pd.DataFrame(
		{"doc_id": new_doc_id, "contents": new_contents, "metadata": new_metadata}
	)
	vectordb_ingest(ingested_vectordb, new_corpus_df, embedding_model)

	assert ingested_vectordb.count() == 8


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_long_text_vectordb_ingest(ingested_vectordb):
	new_doc_id = ["doc6", "doc7"]
	new_contents = ["This is a test" * 20000, "This is a test" * 40000]
	new_metadata = [{"datetime": datetime.now()} for _ in range(2)]
	new_corpus_df = pd.DataFrame(
		{"doc_id": new_doc_id, "contents": new_contents, "metadata": new_metadata}
	)
	assert isinstance(embedding_model, MockEmbedding)
	vectordb_ingest(ingested_vectordb, new_corpus_df, embedding_model)

	assert ingested_vectordb.count() == 7


def mock_get_text_embedding_batch(self, texts, **kwargs):
	return [[3.0, 4.1, 3.2] for _ in range(len(texts))]


@patch.object(
	OpenAIEmbedding, "get_text_embedding_batch", mock_get_text_embedding_batch
)
def test_long_ids_ingest(empty_chromadb):
	embedding_model = OpenAIEmbedding()
	content_df = pd.DataFrame(
		{
			"doc_id": [str(uuid.uuid4()) for _ in range(10000)],
			"contents": ["havertz" for _ in range(10000)],
			"metadata": [
				{"last_modified_datetime": datetime.now()} for _ in range(10000)
			],
		}
	)
	vectordb_ingest(empty_chromadb, content_df, embedding_model)


def test_get_id_scores(ingested_vectordb):
	ids = ["doc2", "doc3", "doc4"]
	embedding_model = MockEmbedding(1536)
	queries = [
		"다이노스 오! 권희동~ 엔씨 오 권희동 오 권희동 권희동 안타~",
		"두산의 헨리 라모스 오오오 라모스 시원하게 화끈하게 날려버려라",
	]
	query_embeddings = embedding_model.get_text_embedding_batch(queries)
	client = chromadb.Client()
	scores = get_id_scores(ids, query_embeddings, ingested_vectordb, client)
	assert len(scores) == 3
	assert isinstance(scores[0], float)
