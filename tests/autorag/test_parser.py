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
simple_data_glob = os.path.join(resource_dir, "parse_data", "all_files", "*")
full_data_glob = os.path.join(resource_dir, "parse_data", "all_files_full", "*")


@pytest.fixture
def simple_parser():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		simple_parser = Parser(data_path_glob=simple_data_glob, project_dir=project_dir)
		yield simple_parser


@pytest.fixture
def test_simple_parser():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		simple_parser = Parser(data_path_glob=simple_data_glob, project_dir=project_dir)
		yield simple_parser


@pytest.fixture
def full_parser():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		full_parser = Parser(data_path_glob=full_data_glob, project_dir=project_dir)
		yield full_parser


@pytest.fixture
def test_full_parser():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		full_parser = Parser(data_path_glob=full_data_glob, project_dir=project_dir)
		yield full_parser


def base_simple_yaml(parser, yaml_path):
	parser.start_parsing(yaml_path, all_files=False)
	project_dir = parser.project_dir
	assert os.path.exists(project_dir)
	assert os.path.exists(os.path.join(project_dir, "parse_config.yaml"))
	assert os.path.exists(os.path.join(project_dir, "pdf.parquet"))
	pdf_result = pd.read_parquet(os.path.join(project_dir, "pdf.parquet"))
	assert os.path.exists(os.path.join(project_dir, "csv.parquet"))
	assert os.path.exists(os.path.join(project_dir, "parsed_result.parquet"))
	parsed_result = pd.read_parquet(os.path.join(project_dir, "parsed_result.parquet"))

	expect_result_columns = ["texts", "path", "page", "last_modified_datetime"]
	assert all(
		[expect_column in pdf_result.columns for expect_column in expect_result_columns]
	)
	expect_result_columns = ["texts", "path", "page", "last_modified_datetime"]
	assert all(
		[
			expect_column in parsed_result.columns
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


def test_start_parsing_specific_type_with_perfect_simple_yaml(simple_parser):
	base_simple_yaml(
		simple_parser,
		os.path.join(resource_dir, "parse_data", "config", "perfect_simple_parse.yaml"),
	)


def test_start_parser_specific_type_with_lack_simple_yaml(simple_parser):
	base_simple_yaml(
		simple_parser,
		os.path.join(resource_dir, "parse_data", "config", "lack_simple_parse.yaml"),
	)


def base_full_yaml(parser, yaml_path):
	parser.start_parsing(yaml_path, all_files=False)
	project_dir = parser.project_dir
	assert os.path.exists(project_dir)
	assert os.path.exists(os.path.join(project_dir, "parse_config.yaml"))
	assert os.path.exists(os.path.join(project_dir, "pdf.parquet"))
	pdf_result = pd.read_parquet(os.path.join(project_dir, "pdf.parquet"))
	assert os.path.exists(os.path.join(project_dir, "csv.parquet"))
	assert os.path.exists(os.path.join(project_dir, "json.parquet"))
	assert os.path.exists(os.path.join(project_dir, "md.parquet"))
	assert os.path.exists(os.path.join(project_dir, "html.parquet"))
	assert os.path.exists(os.path.join(project_dir, "xml.parquet"))
	assert os.path.exists(os.path.join(project_dir, "parsed_result.parquet"))
	parsed_result = pd.read_parquet(os.path.join(project_dir, "parsed_result.parquet"))

	expect_result_columns = ["texts", "path", "page", "last_modified_datetime"]
	assert all(
		[expect_column in pdf_result.columns for expect_column in expect_result_columns]
	)
	expect_result_columns = ["texts", "path", "page", "last_modified_datetime"]
	assert all(
		[
			expect_column in parsed_result.columns
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


def test_start_parsing_specific_type_with_perfect_full_yaml(full_parser):
	base_full_yaml(
		full_parser,
		os.path.join(resource_dir, "parse_data", "config", "perfect_full_parse.yaml"),
	)


def test_start_parsing_specific_type_with_lack_full_yaml(full_parser):
	base_full_yaml(
		full_parser,
		os.path.join(resource_dir, "parse_data", "config", "lack_full_parse.yaml"),
	)


def test_start_parsing_all_files(simple_parser):
	simple_parser.start_parsing(
		os.path.join(resource_dir, "parse_data", "config", "all_files.yaml"),
		all_files=True,
	)
	project_dir = simple_parser.project_dir
	assert os.path.exists(project_dir)
	assert os.path.exists(os.path.join(project_dir, "parse_config.yaml"))
	assert os.path.exists(os.path.join(project_dir, "0.parquet"))
	all_files_result = pd.read_parquet(os.path.join(project_dir, "0.parquet"))

	expect_result_columns = ["texts", "path", "page", "last_modified_datetime"]
	assert all(
		[
			expect_column in all_files_result.columns
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
