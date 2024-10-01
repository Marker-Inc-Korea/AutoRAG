import os
import tempfile
from autorag.data.chunk import llama_index_chunk
from autorag.data.chunk.run import run_chunker

from tests.autorag.data.chunk.test_chunk_base import parsed_result


def test_run_chunker():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
		modules = [llama_index_chunk]
		module_params = [{"chunk_method": "token"}]
		summary_df = run_chunker(modules, module_params, parsed_result, temp_dir)
		assert os.path.exists(os.path.join(temp_dir, "summary.csv"))
		expect_columns = {"filename", "module_name", "module_params", "execution_time"}
		assert set(summary_df.columns) == expect_columns
		assert len(summary_df) == 1
		assert summary_df["module_params"][0] == {"chunk_method": "token"}
		assert os.path.exists(os.path.join(temp_dir, "0.parquet"))
