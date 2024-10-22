import asyncio
import os
import pathlib
import shutil
import tempfile
import uuid
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.nodes.retrieval import VectorDB
from autorag.nodes.retrieval.vectordb import (
	vectordb_ingest,
	get_id_scores,
	filter_exist_ids_from_retrieval_gt,
	filter_exist_ids,
)
from autorag.vectordb.chroma import Chroma
from tests.autorag.nodes.retrieval.test_retrieval_base import (
	queries,
	corpus_df,
	previous_result,
	base_retrieval_test,
	base_retrieval_node_test,
	searchable_input_ids,
)
from tests.mock import mock_aget_text_embedding_batch

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
resource_path = os.path.join(root_dir, "resources")


@pytest.fixture
def mock_chroma():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as chroma_path:
		chroma = Chroma(
			client_type="persistent",
			path=chroma_path,
			embedding_model="mock",
			collection_name="test_vectordb_retrieval",
			similarity_metric="cosine",
		)
		yield chroma


@pytest.fixture
def openai_chroma():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as chroma_path:
		chroma = Chroma(
			client_type="persistent",
			path=chroma_path,
			embedding_model="openai",
			collection_name="test_vectordb_retrieval",
			similarity_metric="cosine",
		)
		yield chroma


@pytest.fixture
def project_dir_for_vectordb_node():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as test_project_dir:
		os.makedirs(os.path.join(test_project_dir, "resources"))
		chroma_path = os.path.join(test_project_dir, "resources", "chroma")
		os.makedirs(chroma_path)

		chroma_config = {
			"client_type": "persistent",
			"path": chroma_path,
			"embedding_model": "mock",
			"collection_name": "mock",
			"similarity_metric": "cosine",
		}
		vectordb_config_path = os.path.join(
			test_project_dir, "resources", "vectordb.yaml"
		)
		with open(vectordb_config_path, "w") as f:
			yaml.safe_dump(
				{"vectordb": [{"name": "mock", "db_type": "chroma", **chroma_config}]},
				f,
			)

		chroma = Chroma(
			**chroma_config,
		)
		os.makedirs(os.path.join(test_project_dir, "data"))
		corpus_path = os.path.join(test_project_dir, "data", "corpus.parquet")
		corpus_df.to_parquet(corpus_path, index=False)
		asyncio.run(vectordb_ingest(chroma, corpus_df))
		yield test_project_dir


@pytest.fixture
def project_dir_for_vectordb_node_from_sample_project():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as test_project_dir:
		sample_project_dir = os.path.join(resource_path, "sample_project")
		# copy & paste all folders and files in the sample_project folder
		shutil.copytree(sample_project_dir, test_project_dir, dirs_exist_ok=True)

		chroma_path = os.path.join(test_project_dir, "resources", "chroma")
		os.makedirs(chroma_path)
		chroma_config = {
			"client_type": "persistent",
			"path": chroma_path,
			"embedding_model": "mock",
			"collection_name": "mock",
			"similarity_metric": "cosine",
		}
		vectordb_config_path = os.path.join(
			test_project_dir, "resources", "vectordb.yaml"
		)
		with open(vectordb_config_path, "w") as f:
			yaml.safe_dump(
				{"vectordb": [{"name": "mock", "db_type": "chroma", **chroma_config}]},
				f,
			)

		chroma = Chroma(**chroma_config)
		corpus_path = os.path.join(test_project_dir, "data", "corpus.parquet")
		local_corpus_df = pd.read_parquet(corpus_path, engine="pyarrow")
		asyncio.run(vectordb_ingest(chroma, local_corpus_df))

		yield test_project_dir


@pytest.fixture
def vectordb_instance(project_dir_for_vectordb_node):
	vectordb = VectorDB(
		project_dir=project_dir_for_vectordb_node,
		vectordb="mock",
	)
	yield vectordb


def test_vectordb_retrieval(vectordb_instance):
	top_k = 4
	id_result, score_result = vectordb_instance._pure(
		queries,
		top_k=top_k,
	)
	base_retrieval_test(id_result, score_result, top_k)


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


def test_vectordb_node(project_dir_for_vectordb_node_from_sample_project):
	result_df = VectorDB.run_evaluator(
		project_dir=project_dir_for_vectordb_node_from_sample_project,
		previous_result=previous_result,
		top_k=4,
		vectordb="mock",
	)
	base_retrieval_node_test(result_df)


def test_vectordb_node_ids(project_dir_for_vectordb_node_from_sample_project):
	result_df = VectorDB.run_evaluator(
		project_dir=project_dir_for_vectordb_node_from_sample_project,
		previous_result=previous_result,
		top_k=4,
		vectordb="mock",
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
	"aget_text_embedding_batch",
	mock_aget_text_embedding_batch,
)
@pytest.mark.asyncio
async def test_duplicate_id_vectordb_ingest(openai_chroma):
	await vectordb_ingest(openai_chroma, corpus_df)
	assert openai_chroma.collection.count() == 5

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
	await vectordb_ingest(openai_chroma, new_corpus_df)

	assert openai_chroma.collection.count() == 8


@patch.object(
	OpenAIEmbedding,
	"aget_text_embedding_batch",
	mock_aget_text_embedding_batch,
)
@pytest.mark.asyncio
async def test_long_text_vectordb_ingest(openai_chroma):
	await vectordb_ingest(openai_chroma, corpus_df)
	new_doc_id = ["doc6", "doc7"]
	new_contents = ["This is a test" * 20000, "This is a test" * 40000]
	new_metadata = [{"datetime": datetime.now()} for _ in range(2)]
	new_corpus_df = pd.DataFrame(
		{"doc_id": new_doc_id, "contents": new_contents, "metadata": new_metadata}
	)
	await vectordb_ingest(openai_chroma, new_corpus_df)

	assert openai_chroma.collection.count() == 7


def mock_get_text_embedding_batch(self, texts, **kwargs):
	return [[3.0, 4.1, 3.2] for _ in range(len(texts))]


@patch.object(
	OpenAIEmbedding, "aget_text_embedding_batch", mock_aget_text_embedding_batch
)
@pytest.mark.asyncio
async def test_long_ids_ingest(openai_chroma):
	content_df = pd.DataFrame(
		{
			"doc_id": [str(uuid.uuid4()) for _ in range(10000)],
			"contents": ["havertz" for _ in range(10000)],
			"metadata": [
				{"last_modified_datetime": datetime.now()} for _ in range(10000)
			],
		}
	)
	await vectordb_ingest(openai_chroma, content_df)


@pytest.mark.asyncio
async def test_filter_exist_ids_from_retrieval_gt(mock_chroma):
	last_modified_datetime = datetime.now()
	ingested_df = pd.DataFrame(
		{
			"doc_id": ["id2"],
			"contents": ["content2"],
			"metadata": [{"last_modified_datetime": last_modified_datetime}],
		}
	)
	await vectordb_ingest(mock_chroma, ingested_df)

	# Create sample qa_data and corpus_data
	qa_data = pd.DataFrame(
		{
			"qid": ["qid1"],
			"query": ["query1"],
			"retrieval_gt": [[["id1", "id2"], ["id3"]]],
			"generation_gt": [["jaxjax"]],
		}
	)
	corpus_data = pd.DataFrame(
		{
			"doc_id": ["id1", "id2", "id3", "id4"],
			"contents": ["content1", "content2", "content3", "content4"],
			"metadata": [
				{"last_modified_datetime": last_modified_datetime} for _ in range(4)
			],
		}
	)

	# Call the function
	result = await filter_exist_ids_from_retrieval_gt(mock_chroma, qa_data, corpus_data)

	# Expected result
	expected_result = pd.DataFrame(
		{
			"doc_id": ["id1", "id3"],
			"contents": ["content1", "content3"],
			"metadata": [
				{
					"last_modified_datetime": last_modified_datetime,
					"prev_id": None,
					"next_id": None,
				}
				for _ in range(2)
			],
		}
	)

	# Assert the result
	pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_result)


@pytest.mark.asyncio
async def test_filter_exist_ids(mock_chroma):
	last_modified_datetime = datetime.now()
	ingested_df = pd.DataFrame(
		{
			"doc_id": ["id2"],
			"contents": ["content2"],
			"metadata": [{"last_modified_datetime": last_modified_datetime}],
		}
	)
	await vectordb_ingest(mock_chroma, ingested_df)

	corpus_data = pd.DataFrame(
		{
			"doc_id": ["id1", "id2", "id3", "id4"],
			"contents": ["content1", "content2", "content3", "content4"],
			"metadata": [
				{"last_modified_datetime": last_modified_datetime} for _ in range(4)
			],
		}
	)

	result = await filter_exist_ids(mock_chroma, corpus_data)

	expected_result = pd.DataFrame(
		{
			"doc_id": ["id1", "id3", "id4"],
			"contents": ["content1", "content3", "content4"],
			"metadata": [
				{
					"last_modified_datetime": last_modified_datetime,
					"prev_id": None,
					"next_id": None,
				}
				for _ in range(3)
			],
		}
	)
	pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_result)


def test_get_id_scores():
	query_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
	content_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
	similarity_metric = "cosine"

	scores = get_id_scores(query_embeddings, content_embeddings, similarity_metric)

	assert len(scores) == len(content_embeddings)
	assert all(isinstance(score, float) for score in scores)
	assert scores == pytest.approx([1.0, 1.0, 1.0])

	similarity_metric = "l2"
	scores = get_id_scores(query_embeddings, content_embeddings, similarity_metric)
	assert len(scores) == len(content_embeddings)
	assert all(isinstance(score, float) for score in scores)
	assert scores == pytest.approx(
		[1.0, 1.0, 1.0]
	)  # Assuming zero distance for identical vectors

	# Test for inner product
	similarity_metric = "ip"
	scores = get_id_scores(query_embeddings, content_embeddings, similarity_metric)
	assert len(scores) == len(content_embeddings)
	assert all(isinstance(score, float) for score in scores)
	assert scores == pytest.approx([0.5, 1.22, 1.94])
