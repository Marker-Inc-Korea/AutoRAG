import os
import pathlib
from typing import Tuple, List, Union, Dict

import pandas as pd

from autorag.evaluation import evaluate_retrieval
from autorag.schema.metricinput import MetricInput
from autorag.strategy import measure_speed, filter_by_threshold, select_best


def evaluate_retrieval_node(
	result_df: pd.DataFrame,
	metric_inputs: List[MetricInput],
	metrics: Union[List[str], List[Dict]],
) -> pd.DataFrame:
	"""
	Evaluate retrieval node from retrieval node result dataframe.
	:param result_df: The result dataframe from a retrieval node.
	:param metric_inputs: List of metric input schema for AutoRAG.
	:param metrics: Metric list from input strategies.
	:return: Return result_df with metrics columns.
	    The columns will be 'retrieved_contents', 'retrieved_ids', 'retrieve_scores', and metric names.
	"""

	@evaluate_retrieval(
		metric_inputs=metric_inputs,
		metrics=metrics,
	)
	def evaluate_this_module(df: pd.DataFrame):
		return (
			df["retrieved_contents"].tolist(),
			df["retrieved_ids"].tolist(),
			df["retrieve_scores"].tolist(),
		)

	return evaluate_this_module(result_df)


def run(
	input_modules,
	input_module_params,
	project_dir: Union[str, pathlib.Path, pathlib.PurePath],
	previous_result: pd.DataFrame,
	strategies,
	metric_inputs: List[MetricInput],
) -> Tuple[List[pd.DataFrame], List]:
	"""
	Run input modules and parameters.
	:param input_modules: Input modules
	:param input_module_params: Input module parameters
	:param project_dir: Project directory path.
	:param previous_result: Previous result dataframe.
	:param strategies: Strategies for retrieval node.
	:param metric_inputs: List of metric input schema for AutoRAG.
	:return: First, it returns list of result dataframe.
	Second, it returns list of execution times.
	"""
	result, execution_times = zip(
		*map(
			lambda task: measure_speed(
				task[0].run_evaluator,
				project_dir=project_dir,
				previous_result=previous_result,
				**task[1],
			),
			zip(input_modules, input_module_params),
		)
	)
	average_times = list(map(lambda x: x / len(result[0]), execution_times))

	# run metrics before filtering
	if strategies.get("metrics") is None:
		raise ValueError("You must at least one metrics for retrieval evaluation.")
	result = list(
		map(
			lambda x: evaluate_retrieval_node(
				x,
				metric_inputs,
				strategies.get("metrics"),
			),
			result,
		)
	)

	return result, average_times


def save_and_summary(
	input_modules,
	input_module_params,
	result_list,
	execution_time_list,
	filename_start: int,
	save_dir: Union[str, pathlib.Path, pathlib.PurePath],
	strategies,
):
	"""
	Save the result and make summary file
	:param input_modules: Input modules
	:param input_module_params: Input module parameters
	:param result_list: Result list
	:param execution_time_list: Execution times
	:param filename_start: The first filename to use
	:return: First, it returns list of result dataframe.
	Second, it returns list of execution times.
	"""

	# save results to folder
	filepaths = list(
		map(
			lambda x: os.path.join(save_dir, f"{x}.parquet"),
			range(filename_start, filename_start + len(input_modules)),
		)
	)
	list(
		map(
			lambda x: x[0].to_parquet(x[1], index=False),
			zip(result_list, filepaths),
		)
	)  # execute save to parquet
	filename_list = list(map(lambda x: os.path.basename(x), filepaths))

	summary_df = pd.DataFrame(
		{
			"filename": filename_list,
			"module_name": list(map(lambda module: module.__name__, input_modules)),
			"module_params": input_module_params,
			"execution_time": execution_time_list,
			**{
				metric: list(map(lambda result: result[metric].mean(), result_list))
				for metric in strategies.get("metrics")
			},
		}
	)
	summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
	return summary_df


def find_best(results, average_times, filenames, strategies):
	# filter by strategies
	if strategies.get("speed_threshold") is not None:
		results, filenames = filter_by_threshold(
			results, average_times, strategies["speed_threshold"], filenames
		)
	selected_result, selected_filename = select_best(
		results,
		strategies.get("metrics"),
		filenames,
		strategies.get("strategy", "mean"),
	)
	return selected_result, selected_filename
