import json
import os.path
from unittest.mock import patch

import autorag
from autorag.data.parse import clova_ocr
from autorag.data.parse.clova import (
	pdf_to_images,
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
	return f"Mocked OCR result for {image_name}", image_name


def check_clova_result(texts, file_names, file_type):
	if file_type == "single_pdf":
		assert texts == [
			"Mocked OCR result for korean_texts_two_page_1.png",
			"Mocked OCR result for korean_texts_two_page_2.png",
		]
		assert file_names == [
			"korean_texts_two_page_1.png",
			"korean_texts_two_page_2.png",
		]
	elif file_type == "multiple_pdf":
		assert texts == [
			"Mocked OCR result for baseball_1_1.png",
			"Mocked OCR result for baseball_2_1.png",
		]
		assert file_names == ["baseball_1_1.png", "baseball_2_1.png"]


@patch.object(autorag.data.parse.clova, "clova_ocr_pure", mock_clova_ocr_pure)
def test_clova_ocr_single_pdf():
	clova_ocr_original = clova_ocr.__wrapped__
	texts, file_names = clova_ocr_original(
		single_pdf_path_list, url="mock_url", api_key="mock_api_key"
	)
	check_clova_result(texts, file_names, "single_pdf")


@patch.object(autorag.data.parse.clova, "clova_ocr_pure", mock_clova_ocr_pure)
def test_clova_ocr_single_pdf_node():
	result_df = clova_ocr(korean_text_glob, url="mock_url", api_key="mock_api_key")
	check_clova_result(
		result_df["texts"].tolist(), result_df["file_name"].tolist(), "single_pdf"
	)


@patch.object(autorag.data.parse.clova, "clova_ocr_pure", mock_clova_ocr_pure)
def test_clova_ocr_multiple_pdf():
	clova_ocr_original = clova_ocr.__wrapped__
	texts, file_names = clova_ocr_original(
		multiple_pdf_data_list, url="mock_url", api_key="mock_api_key"
	)
	check_clova_result(texts, file_names, "multiple_pdf")


@patch.object(autorag.data.parse.clova, "clova_ocr_pure", mock_clova_ocr_pure)
def test_clova_ocr_multiple_pdf_node():
	result_df = clova_ocr(eng_text_glob, url="mock_url", api_key="mock_api_key")
	check_clova_result(
		result_df["texts"].tolist(), result_df["file_name"].tolist(), "multiple_pdf"
	)


def test_pdf_to_images():
	data, names = pdf_to_images(single_pdf_path_list[0])
	assert names == ["korean_texts_two_page_1.png", "korean_texts_two_page_2.png"]


def test_extract_text_from_fields():
	result_path = os.path.join(data_dir, "clova_data", "result_sample.json")
	with open(result_path, "r", encoding="utf-8") as file:
		result = json.load(file)

	text_path = os.path.join(data_dir, "clova_data", "result_text.txt")
	with open(text_path, "r", encoding="utf-8") as file:
		expect_text = file.read()

	text = extract_text_from_fields(result["images"][0]["fields"])
	assert text == expect_text


def test_json_to_html_table():
	result_path = os.path.join(data_dir, "clova_data", "result_sample.json")
	with open(result_path, "r", encoding="utf-8") as file:
		result = json.load(file)

	table_path = os.path.join(data_dir, "clova_data", "result_table.txt")
	with open(table_path, "r", encoding="utf-8") as file:
		expect_table = file.read()

	table = json_to_html_table(result["images"][0]["tables"][0]["cells"])
	assert table == expect_table
