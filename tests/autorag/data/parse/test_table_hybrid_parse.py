import os.path
import tempfile

from autorag.data.parse.table_hybrid_parse import table_hybrid_parse
from autorag.data.parse.table_hybrid_parse import save_page_by_table, get_page_from_path

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
	texts, path, pages = table_hybrid_parse_original(
		hybrid_data_list, **table_hybrid_params
	)
	check_parse_result(texts, path, "hybrid")


def test_table_hybrid_parse_node():
	result_df = table_hybrid_parse(hybrid_glob, **table_hybrid_params)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"hybrid",
	)


def test_table_hybrid_parse_only_text():
	table_hybrid_parse_original = table_hybrid_parse.__wrapped__
	texts, path, pages = table_hybrid_parse_original(
		multiple_pdf_data_list, **table_hybrid_params
	)
	check_parse_result(texts, path, "hybrid_text")


def test_table_hybrid_parse_only_text_node():
	result_df = table_hybrid_parse(eng_text_glob, **table_hybrid_params)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"hybrid_text",
	)


def test_table_hybrid_parse_only_table():
	table_hybrid_parse_original = table_hybrid_parse.__wrapped__
	texts, path, pages = table_hybrid_parse_original(
		table_data_list, **table_hybrid_params
	)
	check_parse_result(texts, path, "hybrid_table")


def test_table_hybrid_parse_only_table_node():
	result_df = table_hybrid_parse(korean_table_glob, **table_hybrid_params)
	check_parse_result(
		result_df["texts"].tolist(),
		result_df["path"].tolist(),
		"hybrid_table",
	)


def test_save_page_by_table():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as save_dir:
		text_dir = os.path.join(save_dir, "text")
		table_dir = os.path.join(save_dir, "table")
		os.makedirs(text_dir, exist_ok=True)
		os.makedirs(table_dir, exist_ok=True)

		path_map_dict = save_page_by_table(hybrid_data_list[0], text_dir, table_dir)
		text_page_dir = os.path.join(text_dir, "nfl_rulebook_both_page_1.pdf")
		table_page_dir = os.path.join(table_dir, "nfl_rulebook_both_page_2.pdf")
		assert os.path.exists(text_dir)
		assert os.path.exists(text_page_dir)
		assert os.path.exists(table_dir)
		assert os.path.exists(table_page_dir)
		assert path_map_dict == {
			text_page_dir: hybrid_data_list[0],
			table_page_dir: hybrid_data_list[0],
		}


def test_split_name_page():
	file_path = "havertz/jax/jeffrey_love_story_page_1.pdf"
	page_num = get_page_from_path(file_path)
	assert page_num == 1


def test_split_name_page_two_page():
	file_path = "havertz/jax/jeffrey_love_story_bali_page_page_2.pdf"
	page_num = get_page_from_path(file_path)
	assert page_num == 2
