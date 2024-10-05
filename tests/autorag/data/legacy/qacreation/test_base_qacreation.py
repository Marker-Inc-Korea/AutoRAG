import os
import pathlib
import re
import tempfile
from typing import Any
from unittest.mock import patch

import chromadb
import pandas as pd
import pytest
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.data.legacy.qacreation import (
	make_single_content_qa,
	generate_qa_llama_index,
	make_qa_with_existing_qa,
	generate_answers,
)
from autorag.utils import validate_qa_dataset
from tests.delete_tests import is_github_action
from tests.mock import MockLLM, mock_get_text_embedding_batch

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent.parent
resource_dir = os.path.join(root_dir, "resources")


@pytest.fixture
def qa_parquet_filepath():
	with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
		yield f.name
		f.close()
		os.unlink(f.name)


@pytest.fixture
def chroma_persistent_client():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
		client = chromadb.PersistentClient(temp_dir)
		yield client


async def acomplete_qa_creation(self, messages, **kwargs: Any):
	pattern = r"Output with (\d+) QnAs:"
	matches = re.findall(pattern, messages)
	num_questions = int(matches[-1])
	return CompletionResponse(
		text="[Q]: Is this the test question?\n[A]: Yes, this is the test answer."
		* num_questions
	)


@patch.object(
	MockLLM,
	"acomplete",
	acomplete_qa_creation,
)
@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
def test_single_content_qa(qa_parquet_filepath):
	corpus_df = pd.read_parquet(
		os.path.join(resource_dir, "corpus_data_sample.parquet")
	)
	qa_df = make_single_content_qa(
		corpus_df,
		content_size=3,
		qa_creation_func=generate_qa_llama_index,
		output_filepath=qa_parquet_filepath,
		llm=MockLLM(),
		question_num_per_content=2,
		upsert=True,
	)
	validate_qa_dataset(qa_df)
	assert len(qa_df) == qa_df["qid"].nunique()
	assert len(qa_df) == 6
	assert qa_df["retrieval_gt"].tolist()[0] == qa_df["retrieval_gt"].tolist()[1]

	assert all([len(x) == 1 and len(x[0]) == 1 for x in qa_df["retrieval_gt"].tolist()])
	assert all([len(x) == 1 for x in qa_df["generation_gt"].tolist()])


@patch.object(
	MockLLM,
	"acomplete",
	acomplete_qa_creation,
)
@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
def test_single_content_qa_long_cache_batch(qa_parquet_filepath):
	corpus_df = pd.read_parquet(
		os.path.join(resource_dir, "corpus_data_sample.parquet")
	)
	qa_df = make_single_content_qa(
		corpus_df,
		content_size=30,
		qa_creation_func=generate_qa_llama_index,
		output_filepath=qa_parquet_filepath,
		llm=MockLLM(),
		question_num_per_content=2,
		upsert=True,
		cache_batch=2,
	)
	validate_qa_dataset(qa_df)
	assert len(qa_df) == 60
	assert qa_df["retrieval_gt"].tolist()[0] == qa_df["retrieval_gt"].tolist()[1]

	assert all([len(x) == 1 and len(x[0]) == 1 for x in qa_df["retrieval_gt"].tolist()])
	assert all([len(x) == 1 for x in qa_df["generation_gt"].tolist()])


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
def test_make_qa_with_existing_qa_without_gen_gt(qa_parquet_filepath):
	corpus_df = pd.read_parquet(
		os.path.join(resource_dir, "corpus_data_sample.parquet"), engine="pyarrow"
	)
	query_df = pd.read_parquet(
		os.path.join(resource_dir, "qa_data_sample.parquet"), engine="pyarrow"
	)
	qa_df = make_qa_with_existing_qa(
		corpus_df,
		query_df,
		content_size=5,
		answer_creation_func=generate_answers,
		output_filepath=qa_parquet_filepath,
		llm=MockLLM(),
		upsert=True,
	)
	validate_qa_dataset(qa_df)
	assert len(qa_df) == 5
	assert all(
		len(elem) == 3 for elem in qa_df["retrieval_gt"].apply(lambda x: x[0]).tolist()
	)
	assert all((elem in query_df["query"].tolist()) for elem in qa_df["query"].tolist())


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
def test_make_qa_with_existing_qa_with_gen_gt(qa_parquet_filepath):
	corpus_df = pd.read_parquet(
		os.path.join(resource_dir, "corpus_data_sample.parquet"), engine="pyarrow"
	)
	query_df = pd.read_parquet(
		os.path.join(resource_dir, "qa_data_sample.parquet"), engine="pyarrow"
	)
	qa_df = make_qa_with_existing_qa(
		corpus_df,
		query_df,
		content_size=5,
		exist_gen_gt=True,
		output_filepath=qa_parquet_filepath,
		upsert=True,
	)
	validate_qa_dataset(qa_df)
	assert len(qa_df) == 5
	assert all(
		len(elem) == 3 for elem in qa_df["retrieval_gt"].apply(lambda x: x[0]).tolist()
	)
	assert all((elem in query_df["query"].tolist()) for elem in qa_df["query"].tolist())
	assert all(
		(elem in query_df["generation_gt"].tolist())
		for elem in qa_df["generation_gt"].tolist()
	)


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
def test_make_qa_with_existing_qa_persistent_client_without_gen_gt(
	chroma_persistent_client, qa_parquet_filepath
):
	corpus_df = pd.read_parquet(
		os.path.join(resource_dir, "corpus_data_sample.parquet"), engine="pyarrow"
	)
	query_df = pd.read_parquet(
		os.path.join(resource_dir, "qa_data_sample.parquet"), engine="pyarrow"
	)
	collection = chroma_persistent_client.get_or_create_collection("auto-rag")
	qa_df = make_qa_with_existing_qa(
		corpus_df,
		query_df,
		content_size=5,
		answer_creation_func=generate_answers,
		output_filepath=qa_parquet_filepath,
		llm=MockLLM(),
		upsert=True,
		collection=collection,
		embedding_model="openai_embed_3_small",
	)
	validate_qa_dataset(qa_df)
	assert len(qa_df) == 5
	assert all(
		len(elem) == 3 for elem in qa_df["retrieval_gt"].apply(lambda x: x[0]).tolist()
	)
	assert all((elem in query_df["query"].tolist()) for elem in qa_df["query"].tolist())


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
def test_make_qa_with_existing_qa_persistent_client_with_gen_gt(
	chroma_persistent_client, qa_parquet_filepath
):
	corpus_df = pd.read_parquet(
		os.path.join(resource_dir, "corpus_data_sample.parquet"), engine="pyarrow"
	)
	query_df = pd.read_parquet(
		os.path.join(resource_dir, "qa_data_sample.parquet"), engine="pyarrow"
	)
	collection = chroma_persistent_client.get_or_create_collection("auto-rag")
	qa_df = make_qa_with_existing_qa(
		corpus_df,
		query_df,
		content_size=5,
		exist_gen_gt=True,
		output_filepath=qa_parquet_filepath,
		upsert=True,
		collection=collection,
		embedding_model="openai_embed_3_small",
	)
	validate_qa_dataset(qa_df)
	assert len(qa_df) == 5
	assert all(
		len(elem) == 3 for elem in qa_df["retrieval_gt"].apply(lambda x: x[0]).tolist()
	)
	assert all((elem in query_df["query"].tolist()) for elem in qa_df["query"].tolist())
	assert all(
		(elem in query_df["generation_gt"].tolist())
		for elem in qa_df["generation_gt"].tolist()
	)
