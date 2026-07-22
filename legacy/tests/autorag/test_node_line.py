from types import SimpleNamespace

import pandas as pd
import pytest

from autorag.node_line import run_node_line


def test_run_node_line_reports_missing_best_module(tmp_path):
	node_type = "generator"
	node_dir = tmp_path / node_type
	node_dir.mkdir()
	pd.DataFrame(
		{
			"is_best": [False],
			"filename": ["0.parquet"],
			"module_name": ["mock"],
			"module_params": [{}],
			"execution_time": [0.1],
		}
	).to_csv(node_dir / "summary.csv", index=False)

	node = SimpleNamespace(
		node_type=node_type,
		run=lambda previous_result, node_line_dir: previous_result,
	)
	previous_result = pd.DataFrame({"query": ["test"]})

	with pytest.raises(ValueError, match="No best module found for node type generator"):
		run_node_line([node], str(tmp_path), previous_result)
