import pytest

from autorag.data import chunk_modules, sentence_splitter_modules
from autorag.data.chunk import llama_index_chunk

from tests.autorag.data.chunk.test_chunk_base import (
	base_texts,
	parsed_result,
	check_chunk_result,
	check_chunk_result_node,
	base_metadata,
	expect_texts,
	expect_token_path,
	expect_token_idx,
	expect_overlap_idx,
)


@pytest.fixture
def chunk_instance():
	chunk_method = "token"
	kwargs = {"chunk_size": 50, "chunk_overlap": 0}
	chunk_instance = chunk_modules[chunk_method](**kwargs)
	yield chunk_instance


def test_llama_index_chunk(chunk_instance):
	llama_index_chunk_original = llama_index_chunk.__wrapped__
	doc_id, contents, path, start_end_idx, metadata = llama_index_chunk_original(
		base_texts, chunk_instance, metadata_list=base_metadata
	)
	check_chunk_result(doc_id, contents, path, start_end_idx, metadata)
	assert len(contents) == 4
	assert contents == expect_texts["token"]
	assert path == expect_token_path
	assert start_end_idx == expect_token_idx


def test_llama_index_chunk_node():
	result_df = llama_index_chunk(parsed_result, chunk_method="token")
	check_chunk_result_node(result_df)
	assert len(result_df["doc_id"].tolist()) == 9


def test_llama_index_chunk_file_name_ko(chunk_instance):
	llama_index_chunk_original = llama_index_chunk.__wrapped__
	doc_id, contents, path, start_end_idx, metadata = llama_index_chunk_original(
		base_texts,
		chunk_instance,
		file_name_language="ko",
		metadata_list=base_metadata,
	)
	check_chunk_result(doc_id, contents, path, start_end_idx, metadata)
	assert len(contents) == 4
	assert contents == expect_texts["token_ko"]
	assert path == expect_token_path
	assert start_end_idx == expect_token_idx


def test_llama_index_chunk_file_name_ko_node():
	result_df = llama_index_chunk(
		parsed_result, chunk_method="token", add_file_name="ko"
	)
	check_chunk_result_node(result_df)
	assert len(result_df["doc_id"].tolist()) == 9


def test_llama_index_chunk_file_name_eng(chunk_instance):
	llama_index_chunk_original = llama_index_chunk.__wrapped__
	doc_id, contents, path, start_end_idx, metadata = llama_index_chunk_original(
		base_texts,
		chunk_instance,
		file_name_language="en",
		metadata_list=base_metadata,
	)
	check_chunk_result(doc_id, contents, path, start_end_idx, metadata)
	assert len(contents) == 4
	assert contents == expect_texts["token_eng"]
	assert path == expect_token_path
	assert start_end_idx == expect_token_idx


def test_llama_index_chunk_file_name_eng_node():
	result_df = llama_index_chunk(
		parsed_result, chunk_method="token", add_file_name="en"
	)
	check_chunk_result_node(result_df)
	assert len(result_df["doc_id"].tolist()) == 9


def test_llama_index_chunk_file_name_ja(chunk_instance):
	llama_index_chunk_original = llama_index_chunk.__wrapped__
	doc_id, contents, path, start_end_idx, metadata = llama_index_chunk_original(
		base_texts,
		chunk_instance,
		file_name_language="ja",
		metadata_list=base_metadata,
	)
	check_chunk_result(doc_id, contents, path, start_end_idx, metadata)
	assert len(contents) == 4
	assert contents == expect_texts["token_ja"]
	assert path == expect_token_path
	assert start_end_idx == expect_token_idx


def test_llama_index_chunk_file_name_ja_node():
	result_df = llama_index_chunk(
		parsed_result, chunk_method="token", add_file_name="ja"
	)
	check_chunk_result_node(result_df)
	assert len(result_df["doc_id"].tolist()) == 9


@pytest.fixture
def chunk_instance_sentence_splitter():
	chunk_method = "sentencewindow"
	sentence_splitter_func = sentence_splitter_modules["kiwi"]()
	kwargs = {"sentence_splitter": sentence_splitter_func}
	chunk_instance_sentence_splitter = chunk_modules[chunk_method](**kwargs)
	yield chunk_instance_sentence_splitter


def test_llama_index_chunk_sentence(chunk_instance_sentence_splitter):
	llama_index_chunk_original = llama_index_chunk.__wrapped__
	doc_id, contents, path, start_end_idx, metadata = llama_index_chunk_original(
		base_texts, chunk_instance_sentence_splitter, metadata_list=base_metadata
	)
	check_chunk_result(doc_id, contents, path, start_end_idx, metadata)
	assert len(contents) == 2
	assert all("window" in meta.keys() for meta in metadata)
	assert path == [base_metadata[0]["path"], base_metadata[1]["path"]]
	assert start_end_idx == [(0, 168), (0, 165)]


def test_llama_index_chunk_sentence_node():
	result_df = llama_index_chunk(
		parsed_result, chunk_method="sentencewindow", sentence_splitter="kiwi"
	)
	check_chunk_result_node(result_df)
	assert len(result_df["doc_id"].tolist()) == 206
	assert all("window" in meta.keys() for meta in (result_df["metadata"].tolist()))


@pytest.fixture
def chunk_instance_overlap():
	chunk_method = "token"
	kwargs = {"chunk_size": 50, "chunk_overlap": 10}
	chunk_instance_overlap = chunk_modules[chunk_method](**kwargs)
	yield chunk_instance_overlap


def test_llama_index_chunk_overlap(chunk_instance_overlap):
	llama_index_chunk_original = llama_index_chunk.__wrapped__
	doc_id, contents, path, start_end_idx, metadata = llama_index_chunk_original(
		base_texts, chunk_instance_overlap, metadata_list=base_metadata
	)
	check_chunk_result(doc_id, contents, path, start_end_idx, metadata)
	assert len(contents) == 4
	assert contents == expect_texts["overlap"]
	assert path == expect_token_path
	assert start_end_idx == expect_overlap_idx
