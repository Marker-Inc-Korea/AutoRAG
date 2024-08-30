from unittest.mock import patch

import autorag
from autorag.data.parse import llama_parse

from tests.autorag.data.parse.test_parse_base import (
	single_pdf_path_list,
	multiple_pdf_data_list,
	korean_text_glob,
	eng_text_glob,
)


async def mock_llama_parse_pure(data_path: str, parse_instance):
	return "Mocked OCR result", "mock_image_name"


def check_llama_parse_result(texts, file_names, file_type):
	if file_type == "single_pdf":
		assert texts == [
			"Mocked OCR result",
		]
		assert file_names == [
			"mock_image_name",
		]
	elif file_type == "multiple_pdf":
		assert texts == [
			"Mocked OCR result",
			"Mocked OCR result",
		]
		assert file_names == [
			"mock_image_name",
			"mock_image_name",
		]


@patch.object(autorag.data.parse.llamaparse, "llama_parse_pure", mock_llama_parse_pure)
def test_llama_parse_single_pdf():
	llama_parse_original = llama_parse.__wrapped__
	texts, file_names = llama_parse_original(
		single_pdf_path_list, url="mock_url", api_key="mock_api_key"
	)
	check_llama_parse_result(texts, file_names, "single_pdf")


@patch.object(autorag.data.parse.llamaparse, "llama_parse_pure", mock_llama_parse_pure)
def test_llama_parse_single_pdf_node():
	result_df = llama_parse(korean_text_glob, url="mock_url", api_key="mock_api_key")
	check_llama_parse_result(
		result_df["texts"].tolist(), result_df["file_name"].tolist(), "single_pdf"
	)


@patch.object(autorag.data.parse.llamaparse, "llama_parse_pure", mock_llama_parse_pure)
def test_llama_parse_multiple_pdf():
	llama_parse_original = llama_parse.__wrapped__
	texts, file_names = llama_parse_original(
		multiple_pdf_data_list, url="mock_url", api_key="mock_api_key"
	)
	check_llama_parse_result(texts, file_names, "multiple_pdf")


@patch.object(autorag.data.parse.llamaparse, "llama_parse_pure", mock_llama_parse_pure)
def test_llama_parse_multiple_pdf_node():
	result_df = llama_parse(eng_text_glob, url="mock_url", api_key="mock_api_key")
	check_llama_parse_result(
		result_df["texts"].tolist(), result_df["file_name"].tolist(), "multiple_pdf"
	)
