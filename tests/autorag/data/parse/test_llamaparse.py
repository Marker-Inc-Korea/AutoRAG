import os
from typing import List
from unittest.mock import patch

from autorag.data.parse.llamaparse import llama_parse as llamaparse

from tests.autorag.data.parse.test_parse_base import (
	single_pdf_path_list,
	multiple_pdf_data_list,
	korean_text_glob,
	eng_text_glob,
	check_parse_result,
)

from llama_index.core.schema import Document
import llama_parse


async def mock_llama_parse_aload_data(*args, **kwargs) -> List[Document]:
	return [Document(id_="test_id", text="I love AutoRAG", metadata={})]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_single_pdf():
	llama_parse_original = llamaparse.__wrapped__
	texts, path, pages = llama_parse_original(
		single_pdf_path_list, url="mock_url", api_key="mock_api_key"
	)
	check_parse_result(texts, path, "single_pdf")
	assert pages == [1]
	assert texts == ["I love AutoRAG"]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_single_pdf_node():
	result_df = llamaparse(
		korean_text_glob, file_type="all_files", url="mock_url", api_key="mock_api_key"
	)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"single_pdf",
	)
	assert result_df["page"].tolist() == [1]
	assert result_df["texts"].tolist() == ["I love AutoRAG"]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_multiple_pdf():
	llama_parse_original = llamaparse.__wrapped__
	texts, path, pages = llama_parse_original(
		multiple_pdf_data_list, url="mock_url", api_key="mock_api_key"
	)
	check_parse_result(texts, path, "multiple_pdf")
	assert pages == [1, 1]
	assert texts == ["I love AutoRAG", "I love AutoRAG"]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_multiple_pdf_node():
	result_df = llamaparse(
		eng_text_glob, file_type="all_files", url="mock_url", api_key="mock_api_key"
	)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"multiple_pdf",
	)
	assert result_df["page"].tolist() == [1, 1]
	assert result_df["texts"].tolist() == ["I love AutoRAG", "I love AutoRAG"]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_multimodal():
	llama_parse_original = llamaparse.__wrapped__
	texts, path, pages = llama_parse_original(
		multiple_pdf_data_list,
		url="mock_url",
		api_key="mock_api_key",
		use_vendor_multimodal_model=True,
		vendor_multimodal_model_name="openai-gpt-4o-mini",
	)
	check_parse_result(texts, path, "multiple_pdf")
	assert pages == [1, 1]
	assert texts == ["I love AutoRAG", "I love AutoRAG"]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_multimodal_node():
	result_df = llamaparse(
		eng_text_glob,
		file_type="all_files",
		url="mock_url",
		api_key="mock_api_key",
		use_vendor_multimodal_model=True,
		vendor_multimodal_model_name="openai-gpt-4o-mini",
	)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"multiple_pdf",
	)
	assert result_df["page"].tolist() == [1, 1]
	assert result_df["texts"].tolist() == ["I love AutoRAG", "I love AutoRAG"]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_multimodal_use_env_key():
	temp_env_vars = {
		"OPENAI_API_KEY": "mock_openai_api_key",
	}

	with patch.dict(os.environ, temp_env_vars):
		llama_parse_original = llamaparse.__wrapped__
		texts, path, pages = llama_parse_original(
			multiple_pdf_data_list,
			url="mock_url",
			api_key="mock_api_key",
			use_vendor_multimodal_model=True,
			vendor_multimodal_model_name="openai-gpt-4o-mini",
			use_own_key=True,
		)
		check_parse_result(texts, path, "multiple_pdf")
		assert pages == [1, 1]
		assert texts == ["I love AutoRAG", "I love AutoRAG"]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_multimodal_use_env_key_node():
	temp_env_vars = {
		"OPENAI_API_KEY": "mock_openai_api_key",
	}

	with patch.dict(os.environ, temp_env_vars):
		result_df = llamaparse(
			eng_text_glob,
			file_type="all_files",
			url="mock_url",
			api_key="mock_api_key",
			use_vendor_multimodal_model=True,
			vendor_multimodal_model_name="openai-gpt-4o-mini",
			use_own_key=True,
		)
		check_parse_result(
			result_df["texts"].tolist(),
			result_df["path"].tolist(),
			"multiple_pdf",
		)
		assert result_df["page"].tolist() == [1, 1]
		assert result_df["texts"].tolist() == ["I love AutoRAG", "I love AutoRAG"]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_multimodal_use_own_key():
	llama_parse_original = llamaparse.__wrapped__
	texts, path, pages = llama_parse_original(
		multiple_pdf_data_list,
		url="mock_url",
		api_key="mock_api_key",
		use_vendor_multimodal_model=True,
		vendor_multimodal_model_name="openai-gpt-4o-mini",
		vendor_multimodal_api_key="mock_openai_api_key",
	)
	check_parse_result(texts, path, "multiple_pdf")
	assert pages == [1, 1]
	assert texts == ["I love AutoRAG", "I love AutoRAG"]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_multimodal_use_own_key_node():
	result_df = llamaparse(
		eng_text_glob,
		file_type="all_files",
		url="mock_url",
		api_key="mock_api_key",
		use_vendor_multimodal_model=True,
		vendor_multimodal_model_name="openai-gpt-4o-mini",
		use_own_key=True,
		vendor_multimodal_api_key="mock_openai_api_key",
	)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"multiple_pdf",
	)
	assert result_df["page"].tolist() == [1, 1]
	assert result_df["texts"].tolist() == ["I love AutoRAG", "I love AutoRAG"]
