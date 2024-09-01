import tempfile
import pytest

from autorag.data.parse import table_hybrid_parse

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
