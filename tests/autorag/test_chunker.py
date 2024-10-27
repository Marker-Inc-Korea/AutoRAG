import os
import pathlib
import tempfile

import pandas as pd
import pytest

from autorag.chunker import Chunker
from autorag.utils.util import load_summary_file

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
resource_dir = os.path.join(root_dir, "resources")
data_dir = os.path.join(resource_dir, "chunk_data")
parsed_data_path = os.path.join(data_dir, "sample_parsed.parquet")


@pytest.fixture
def chunker():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		chunker = Chunker.from_parquet(
			parsed_data_path=parsed_data_path, project_dir=project_dir
		)
		yield chunker


@pytest.fixture
def chunker_test_fixture():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		chunker = Chunker.from_parquet(
			parsed_data_path=parsed_data_path, project_dir=project_dir
		)
		yield chunker


def test_start_chunking(chunker):
	chunker.start_chunking(os.path.join(resource_dir, "simple_chunk.yaml"))
	project_dir = chunker.project_dir
	assert os.path.exists(project_dir)
	assert os.path.exists(os.path.join(project_dir, "chunk_config.yaml"))
	assert os.path.exists(os.path.join(project_dir, "0.parquet"))
	chunk_result = pd.read_parquet(os.path.join(project_dir, "0.parquet"))
	expect_result_columns = ["doc_id", "contents", "metadata"]
	assert all(
		[
			expect_column in chunk_result.columns
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
