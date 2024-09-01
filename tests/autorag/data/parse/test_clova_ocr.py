import json
import os.path
from unittest.mock import patch

import pytest

import autorag
from autorag.data.parse import clova_ocr
from autorag.data.parse.clova import (
	pdf_to_images,
	generate_image_names,
	extract_text_from_fields,
	json_to_html_table,
)

from tests.autorag.data.parse.test_parse_base import (
	single_pdf_path_list,
	multiple_pdf_data_list,
	data_dir,
	korean_text_glob,
	eng_text_glob,
)


async def mock_clova_ocr_pure(
	image_data: bytes,
	image_name: str,
	url: str,
	api_key: str,
	table_detection: bool = False,
):
	return "Mocked OCR result", "mock_image_name"


def check_clova_result(texts, file_names):
	assert texts == [
		"Mocked OCR result",
		"Mocked OCR result",
	]
	assert file_names == [
		"mock_image_name",
		"mock_image_name",
	]


@pytest.fixture
def result_sample():
	result_path = os.path.join(data_dir, "clova_data", "result_sample.json")
	with open(result_path, "r", encoding="utf-8") as file:
		result_sample = json.load(file)
		yield result_sample


@pytest.fixture
def expect_text():
	text_path = os.path.join(data_dir, "clova_data", "result_text.txt")
	with open(text_path, "r", encoding="utf-8") as file:
		expect_text = file.read()
		expect_text = expect_text.split("\n\\n\\n\\n")[0]
		yield expect_text


@pytest.fixture
def expect_table():
	table_path = os.path.join(data_dir, "clova_data", "result_table.txt")
	with open(table_path, "r", encoding="utf-8") as file:
		expect_table = file.read()
		expect_table = expect_table.split("\n\\n\\n\\n")[0]
		yield expect_table


@patch.object(autorag.data.parse.clova, "clova_ocr_pure", mock_clova_ocr_pure)
def test_clova_ocr_single_pdf():
	clova_ocr_original = clova_ocr.__wrapped__
	texts, file_names = clova_ocr_original(
		single_pdf_path_list, url="mock_url", api_key="mock_api_key"
	)
	check_clova_result(texts, file_names)


@patch.object(autorag.data.parse.clova, "clova_ocr_pure", mock_clova_ocr_pure)
def test_clova_ocr_single_pdf_node():
	result_df = clova_ocr(korean_text_glob, url="mock_url", api_key="mock_api_key")
	check_clova_result(result_df["texts"].tolist(), result_df["file_name"].tolist())


@patch.object(autorag.data.parse.clova, "clova_ocr_pure", mock_clova_ocr_pure)
def test_clova_ocr_multiple_pdf():
	clova_ocr_original = clova_ocr.__wrapped__
	texts, file_names = clova_ocr_original(
		multiple_pdf_data_list, url="mock_url", api_key="mock_api_key"
	)
	check_clova_result(texts, file_names)


@patch.object(autorag.data.parse.clova, "clova_ocr_pure", mock_clova_ocr_pure)
def test_clova_ocr_multiple_pdf_node():
	result_df = clova_ocr(eng_text_glob, url="mock_url", api_key="mock_api_key")
	check_clova_result(result_df["texts"].tolist(), result_df["file_name"].tolist())


def test_pdf_to_images():
	data_list = pdf_to_images(multiple_pdf_data_list[0])
	assert isinstance(data_list, list)
	assert isinstance(data_list[0], bytes)


def test_generate_image_names():
	names = generate_image_names(single_pdf_path_list[0], 2)
	assert names == ["korean_texts_two_page_1.png", "korean_texts_two_page_2.png"]


def test_extract_text_from_fields(result_sample, expect_text):
	text = extract_text_from_fields(result_sample["images"][0]["fields"])
	assert text == expect_text


def test_json_to_html_table(result_sample, expect_table):
	table = json_to_html_table(result_sample["images"][0]["tables"][0]["cells"])
	assert table == expect_table
