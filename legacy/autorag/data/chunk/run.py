import os
from typing import Callable, List, Dict
import pandas as pd

from autorag.strategy import measure_speed


def run_chunker(
	modules: List[Callable],
	module_params: List[Dict],
	parsed_result: pd.DataFrame,
	project_dir: str,
):
	results, execution_times = zip(
		*map(
			lambda x: measure_speed(x[0], parsed_result=parsed_result, **x[1]),
			zip(modules, module_params),
		)
	)
	average_times = list(map(lambda x: x / len(results[0]), execution_times))

	# save results to parquet files
	filepaths = list(
		map(lambda x: os.path.join(project_dir, f"{x}.parquet"), range(len(modules)))
	)
	list(map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths)))
	filenames = list(map(lambda x: os.path.basename(x), filepaths))

	summary_df = pd.DataFrame(
		{
			"filename": filenames,
			"module_name": list(map(lambda module: module.__name__, modules)),
			"module_params": module_params,
			"execution_time": average_times,
		}
	)
	summary_df.to_csv(os.path.join(project_dir, "summary.csv"), index=False)
	return summary_df
