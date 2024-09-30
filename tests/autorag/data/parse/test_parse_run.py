import os
import tempfile

from autorag.data.parse import langchain_parse
from autorag.data.parse.run import run_parser

from tests.autorag.data.parse.test_parse_base import eng_text_glob


def test_run_parser():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
		modules = [langchain_parse, langchain_parse]
		module_params = [{"parse_method": "pdfminer"}, {"parse_method": "pdfplumber"}]
		data_path_glob = eng_text_glob
		summary_df = run_parser(modules, module_params, data_path_glob, temp_dir)
		assert os.path.exists(os.path.join(temp_dir, "summary.csv"))
		expect_columns = {"filename", "module_name", "module_params", "execution_time"}
		assert set(summary_df.columns) == expect_columns
		assert len(summary_df) == 2
		assert summary_df["module_params"][0] == {"parse_method": "pdfminer"}
		assert os.path.exists(os.path.join(temp_dir, "0.parquet"))
