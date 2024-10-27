import os
import pathlib
import tempfile

import pandas as pd
import pytest

from autorag.parser import Parser
from autorag.utils.util import load_summary_file

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
resource_dir = os.path.join(root_dir, "resources")
result_dir = os.path.join(resource_dir, "test_results")
data_glob = os.path.join(resource_dir, "parse_data", "eng_text", "*")


@pytest.fixture
def parser():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		parser = Parser(data_path_glob=data_glob, project_dir=project_dir)
		yield parser


@pytest.fixture
def test_parser():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		parser = Parser(data_path_glob=data_glob, project_dir=project_dir)
		yield parser


def test_start_parsing(parser):
	parser.start_parsing(os.path.join(resource_dir, "simple_parse.yaml"))
	project_dir = parser.project_dir
	assert os.path.exists(project_dir)
	assert os.path.exists(os.path.join(project_dir, "parse_config.yaml"))
	assert os.path.exists(os.path.join(project_dir, "0.parquet"))
	pdfminer_result = pd.read_parquet(os.path.join(project_dir, "0.parquet"))
	assert os.path.exists(os.path.join(project_dir, "1.parquet"))
	pdfplumber_result = pd.read_parquet(os.path.join(project_dir, "1.parquet"))

	expect_result_columns = ["texts", "path", "page", "last_modified_datetime"]
	assert all(
		[
			expect_column in pdfminer_result.columns
			for expect_column in expect_result_columns
		]
	)
	assert all(
		[
			expect_column in pdfplumber_result.columns
			for expect_column in expect_result_columns
		]
	)
	summary_df = load_summary_file(os.path.join(project_dir, "summary.csv"))
	expect_summary_columns = [
		"filename",
		"module_name",
		"module_params",
		"execution_time",
	]
	assert all(
		[
			expect_column in summary_df.columns
			for expect_column in expect_summary_columns
		]
	)
