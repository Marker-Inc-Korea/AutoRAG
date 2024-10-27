import pytest

from autorag.data import chunk_modules
from autorag.data.chunk import langchain_chunk

from tests.autorag.data.chunk.test_chunk_base import (
	base_texts,
	parsed_result,
	check_chunk_result,
	base_metadata,
	character_expect_texts,
	expect_texts,
	expect_character_idx,
	expect_character_path,
	check_chunk_result_node,
)


@pytest.fixture
def chunk_instance():
	chunk_method = "character"
	kwargs = {"separator": ". ", "chunk_size": 30, "chunk_overlap": 0}
	chunk_instance = chunk_modules[chunk_method](**kwargs)
	yield chunk_instance


def test_langchain_chunk(chunk_instance):
	langchain_chunk_original = langchain_chunk.__wrapped__
	doc_id, contents, path, start_end_idx, metadata = langchain_chunk_original(
		base_texts, chunk_instance, metadata_list=base_metadata
	)
	check_chunk_result(doc_id, contents, path, start_end_idx, metadata)
	assert len(contents) == 7
	assert contents == character_expect_texts
	assert path == expect_character_path
	assert start_end_idx == expect_character_idx


def test_langchain_chunk_node(chunk_instance):
	result_df = langchain_chunk(parsed_result, chunk_method="character")
	check_chunk_result_node(result_df)
	assert len(result_df["doc_id"].tolist()) == 9


def test_langchain_chunk_file_name_ko(chunk_instance):
	langchain_chunk_original = langchain_chunk.__wrapped__
	doc_id, contents, path, start_end_idx, metadata = langchain_chunk_original(
		base_texts,
		chunk_instance,
		file_name_language="ko",
		metadata_list=base_metadata,
	)
	check_chunk_result(doc_id, contents, path, start_end_idx, metadata)
	assert len(contents) == 7
	assert contents == expect_texts["character_ko"]
	assert path == expect_character_path
	assert start_end_idx == expect_character_idx


def test_langchain_chunk_file_name_ko_node(chunk_instance):
	result_df = langchain_chunk(
		parsed_result, chunk_method="character", add_file_name="ko"
	)
	check_chunk_result_node(result_df)
	assert len(result_df["doc_id"].tolist()) == 9


def test_langchain_chunk_file_name_eng(chunk_instance):
	langchain_chunk_original = langchain_chunk.__wrapped__
	doc_id, contents, path, start_end_idx, metadata = langchain_chunk_original(
		base_texts,
		chunk_instance,
		file_name_language="en",
		metadata_list=base_metadata,
	)
	check_chunk_result(doc_id, contents, path, start_end_idx, metadata)
	assert len(contents) == 7
	assert contents == expect_texts["character_eng"]
	assert path == expect_character_path
	assert start_end_idx == expect_character_idx


def test_langchain_chunk_file_name_eng_node():
	result_df = langchain_chunk(
		parsed_result, chunk_method="character", add_file_name="en"
	)
	check_chunk_result_node(result_df)
	assert len(result_df["doc_id"].tolist()) == 9


def test_langchain_chunk_file_name_ja(chunk_instance):
	langchain_chunk_original = langchain_chunk.__wrapped__
	doc_id, contents, path, start_end_idx, metadata = langchain_chunk_original(
		base_texts,
		chunk_instance,
		file_name_language="ja",
		metadata_list=base_metadata,
	)
	check_chunk_result(doc_id, contents, path, start_end_idx, metadata)
	assert len(contents) == 7
	assert contents == expect_texts["character_ja"]
	assert path == expect_character_path
	assert start_end_idx == expect_character_idx


def test_langchain_chunk_file_name_ja_node(chunk_instance):
	result_df = langchain_chunk(
		parsed_result, chunk_method="character", add_file_name="ja"
	)
	check_chunk_result_node(result_df)
	assert len(result_df["doc_id"].tolist()) == 9
