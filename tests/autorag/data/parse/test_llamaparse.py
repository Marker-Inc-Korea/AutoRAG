from typing import List
from unittest.mock import patch

from autorag.data.parse import llama_parse as llamaparse

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
	check_parse_result(texts, path, pages, "single_pdf", "llama")
	assert texts == ["I love AutoRAG"]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_single_pdf_node():
	result_df = llamaparse(korean_text_glob, url="mock_url", api_key="mock_api_key")
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		result_df["page"].tolist(),
		"single_pdf",
		"llama",
	)
	assert result_df["texts"].tolist() == ["I love AutoRAG"]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_multiple_pdf():
	llama_parse_original = llamaparse.__wrapped__
	texts, path, pages = llama_parse_original(
		multiple_pdf_data_list, url="mock_url", api_key="mock_api_key"
	)
	check_parse_result(texts, path, pages, "multiple_pdf", "llama")
	assert texts == ["I love AutoRAG", "I love AutoRAG"]


@patch.object(llama_parse.base.LlamaParse, "aload_data", mock_llama_parse_aload_data)
def test_llama_parse_multiple_pdf_node():
	result_df = llamaparse(eng_text_glob, url="mock_url", api_key="mock_api_key")
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		result_df["page"].tolist(),
		"multiple_pdf",
		"llama",
	)
	assert result_df["texts"].tolist() == ["I love AutoRAG", "I love AutoRAG"]
