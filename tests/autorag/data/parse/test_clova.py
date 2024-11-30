import json
import os.path
from unittest.mock import patch

import pytest

import autorag
from autorag.data.parse.clova import (
	clova_ocr,
	pdf_to_images,
	generate_image_info,
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
	return "Mocked OCR result", single_pdf_path_list[0], 1


def check_clova_result(texts, path, pages):
	assert texts == [
		"Mocked OCR result",
		"Mocked OCR result",
	]
	assert path == [
		single_pdf_path_list[0],
		single_pdf_path_list[0],
	]
	assert pages == [1, 1]


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
	texts, path, pages = clova_ocr_original(
		single_pdf_path_list, url="mock_url", api_key="mock_api_key"
	)
	check_clova_result(texts, path, pages)


@patch.object(autorag.data.parse.clova, "clova_ocr_pure", mock_clova_ocr_pure)
def test_clova_ocr_single_pdf_node():
	result_df = clova_ocr(
		korean_text_glob, file_type="all_files", url="mock_url", api_key="mock_api_key"
	)
	check_clova_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		result_df["page"].tolist(),
	)


@patch.object(autorag.data.parse.clova, "clova_ocr_pure", mock_clova_ocr_pure)
def test_clova_ocr_multiple_pdf():
	clova_ocr_original = clova_ocr.__wrapped__
	texts, path, pages = clova_ocr_original(
		multiple_pdf_data_list, url="mock_url", api_key="mock_api_key"
	)
	check_clova_result(texts, path, pages)


@patch.object(autorag.data.parse.clova, "clova_ocr_pure", mock_clova_ocr_pure)
def test_clova_ocr_multiple_pdf_node():
	result_df = clova_ocr(
		eng_text_glob, file_type="all_files", url="mock_url", api_key="mock_api_key"
	)
	check_clova_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		result_df["page"].tolist(),
	)


def test_pdf_to_images():
	data_list = pdf_to_images(multiple_pdf_data_list[0])
	assert isinstance(data_list, list)
	assert isinstance(data_list[0], bytes)


def test_generate_image_info():
	names = generate_image_info(single_pdf_path_list[0], 2)
	assert names == [
		{"pdf_path": single_pdf_path_list[0], "pdf_page": 1},
		{"pdf_path": single_pdf_path_list[0], "pdf_page": 2},
	]


def test_extract_text_from_fields(result_sample, expect_text):
	text = extract_text_from_fields(result_sample["images"][0]["fields"])
	assert text == expect_text


def test_json_to_html_table(result_sample, expect_table):
	table = json_to_html_table(result_sample["images"][0]["tables"][0]["cells"])
	assert table == expect_table
