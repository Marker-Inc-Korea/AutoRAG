import os.path
import tempfile
import pytest

from autorag.data.parse import table_hybrid_parse
from autorag.data.parse.table_hybrid_parse import save_page_by_table

from tests.autorag.data.parse.test_parse_base import (
	hybrid_data_list,
	hybrid_glob,
	check_parse_result,
)


@pytest.fixture
def table_hybrid_params():
	with tempfile.TemporaryDirectory() as pages_save_dir:
		table_hybrid_params = {
			"text_parse_module": "langchain_parse",
			"text_params": {"parse_method": "pdfminer"},
			"table_parse_module": "langchain_parse",
			"table_params": {"parse_method": "pdfplumber"},
			"pages_save_dir": pages_save_dir,
		}
		yield table_hybrid_params


def test_table_hybrid_parse(table_hybrid_params):
	table_hybrid_parse_original = table_hybrid_parse.__wrapped__
	texts, file_names = table_hybrid_parse_original(
		hybrid_data_list, **table_hybrid_params
	)
	check_parse_result(texts, file_names, "hybrid")


def test_table_hybrid_parse_node(table_hybrid_params):
	result_df = table_hybrid_parse(hybrid_glob, **table_hybrid_params)
	check_parse_result(
		result_df["texts"].tolist(), result_df["file_name"].tolist(), "hybrid"
	)


def test_save_page_by_table(table_hybrid_params):
	save_dir = table_hybrid_params["pages_save_dir"]
	text_dir = os.path.join(save_dir, "text")
	table_dir = os.path.join(save_dir, "table")

	save_page_by_table(hybrid_data_list[0], text_dir, table_dir)
	assert os.path.exists(text_dir)
	assert os.path.exists(os.path.join(text_dir, "nfl_rulebook_both_page_1.pdf"))
	assert os.path.exists(table_dir)
	assert os.path.exists(os.path.join(table_dir, "nfl_rulebook_both_page_2.pdf"))
