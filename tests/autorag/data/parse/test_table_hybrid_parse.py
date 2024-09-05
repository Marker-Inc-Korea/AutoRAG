import os.path
import tempfile

from autorag.data.parse import table_hybrid_parse
from autorag.data.parse.table_hybrid_parse import save_page_by_table, split_name_page

from tests.autorag.data.parse.test_parse_base import (
	hybrid_data_list,
	hybrid_glob,
	check_parse_result,
	multiple_pdf_data_list,
	eng_text_glob,
	table_data_list,
	korean_table_glob,
)

table_hybrid_params = {
	"text_parse_module": "langchain_parse",
	"text_params": {"parse_method": "pdfminer"},
	"table_parse_module": "langchain_parse",
	"table_params": {"parse_method": "pdfplumber"},
}


def test_table_hybrid_parse():
	table_hybrid_parse_original = table_hybrid_parse.__wrapped__
	texts, file_names, pages = table_hybrid_parse_original(
		hybrid_data_list, **table_hybrid_params
	)
	check_parse_result(texts, file_names, pages, "hybrid", "hybrid")


def test_table_hybrid_parse_node():
	result_df = table_hybrid_parse(hybrid_glob, **table_hybrid_params)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["file_name"].tolist(),
		result_df["page"].tolist(),
		"hybrid",
		"hybrid",
	)


def test_table_hybrid_parse_only_text():
	table_hybrid_parse_original = table_hybrid_parse.__wrapped__
	texts, file_names, pages = table_hybrid_parse_original(
		multiple_pdf_data_list, **table_hybrid_params
	)
	check_parse_result(texts, file_names, pages, "hybrid_text", "hybrid")


def test_table_hybrid_parse_only_text_node():
	result_df = table_hybrid_parse(eng_text_glob, **table_hybrid_params)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["file_name"].tolist(),
		result_df["page"].tolist(),
		"hybrid_text",
		"hybrid",
	)


def test_table_hybrid_parse_only_table():
	table_hybrid_parse_original = table_hybrid_parse.__wrapped__
	texts, file_names, pages = table_hybrid_parse_original(
		table_data_list, **table_hybrid_params
	)
	check_parse_result(texts, file_names, pages, "hybrid_table", "hybrid")


def test_table_hybrid_parse_only_table_node():
	result_df = table_hybrid_parse(korean_table_glob, **table_hybrid_params)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["file_name"].tolist(),
		result_df["page"].tolist(),
		"hybrid_table",
		"hybrid",
	)


def test_save_page_by_table():
	with tempfile.TemporaryDirectory() as save_dir:
		text_dir = os.path.join(save_dir, "text")
		table_dir = os.path.join(save_dir, "table")
		os.makedirs(text_dir, exist_ok=True)
		os.makedirs(table_dir, exist_ok=True)

		save_page_by_table(hybrid_data_list[0], text_dir, table_dir)
		assert os.path.exists(text_dir)
		assert os.path.exists(os.path.join(text_dir, "nfl_rulebook_both_page_1.pdf"))
		assert os.path.exists(table_dir)
		assert os.path.exists(os.path.join(table_dir, "nfl_rulebook_both_page_2.pdf"))


def test_split_name_page():
	file_name = "jeffrey_love_story_page_1.pdf"
	pure_file_name, page_num = split_name_page(file_name)
	assert pure_file_name == "jeffrey_love_story.pdf"
	assert page_num == 1


def test_split_name_page_two_page():
	file_name = "jeffrey_love_story_bali_page_page_2.pdf"
	pure_file_name, page_num = split_name_page(file_name)
	assert pure_file_name == "jeffrey_love_story_bali_page.pdf"
	assert page_num == 2
