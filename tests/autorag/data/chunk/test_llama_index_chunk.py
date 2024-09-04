import pytest

from autorag.data import chunk_modules
from autorag.data.chunk import llama_index_chunk

from tests.autorag.data.chunk.test_chunk_base import (
	base_texts,
	parsed_result,
	check_chunk_result,
	base_file_names,
	expect_texts,
)


@pytest.fixture
def chunk_instance():
	chunk_method = "token"
	kwargs = {"chunk_size": 30, "chunk_overlap": 0}
	chunk_instance = chunk_modules[chunk_method](**kwargs)
	yield chunk_instance


def test_llama_index_chunk(chunk_instance):
	llama_index_chunk_original = llama_index_chunk.__wrapped__
	doc_id, contents, metadata = llama_index_chunk_original(base_texts, chunk_instance)
	check_chunk_result(doc_id, metadata)
	assert len(contents) == 4
	assert contents == expect_texts["original"]


def test_llama_index_chunk_node():
	result_df = llama_index_chunk(parsed_result, chunk_method="token")
	check_chunk_result(result_df["doc_id"].tolist(), result_df["metadata"].tolist())
	assert len(result_df["doc_id"].tolist()) == 9


def test_llama_index_chunk_file_name_ko(chunk_instance):
	llama_index_chunk_original = llama_index_chunk.__wrapped__
	doc_id, contents, metadata = llama_index_chunk_original(
		base_texts,
		chunk_instance,
		file_name_language="korean",
		file_names=base_file_names,
	)
	check_chunk_result(doc_id, metadata)
	assert len(contents) == 6
	assert contents == expect_texts["korean"]


def test_llama_index_chunk_file_name_ko_node():
	result_df = llama_index_chunk(
		parsed_result, chunk_method="token", add_file_name="korean"
	)
	check_chunk_result(result_df["doc_id"].tolist(), result_df["metadata"].tolist())
	assert len(result_df["doc_id"].tolist()) == 9


def test_llama_index_chunk_file_name_eng(chunk_instance):
	llama_index_chunk_original = llama_index_chunk.__wrapped__
	doc_id, contents, metadata = llama_index_chunk_original(
		base_texts,
		chunk_instance,
		file_name_language="english",
		file_names=base_file_names,
	)
	check_chunk_result(doc_id, metadata)
	assert len(contents) == 6
	assert contents == expect_texts["english"]


def test_llama_index_chunk_file_name_eng_node():
	result_df = llama_index_chunk(
		parsed_result, chunk_method="token", add_file_name="english"
	)
	check_chunk_result(result_df["doc_id"].tolist(), result_df["metadata"].tolist())
	assert len(result_df["doc_id"].tolist()) == 9
