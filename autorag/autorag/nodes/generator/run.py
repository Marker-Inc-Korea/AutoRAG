import os
import pathlib
from typing import List, Dict, Union

import pandas as pd

from autorag.evaluation import evaluate_generation
from autorag.evaluation.util import cast_metrics
from autorag.schema.metricinput import MetricInput
from autorag.strategy import measure_speed, filter_by_threshold, select_best
from autorag.utils.util import to_list


def run_generator_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run evaluation and select the best module among generator node results.
	And save the results and summary to generator node directory.

	:param modules: Generator modules to run.
	:param module_params: Generator module parameters.
	    Including node parameters, which is used for every module in this node.
	:param previous_result: Previous result dataframe.
	    Could be prompt maker node's result.
	:param node_line_dir: This node line's directory.
	:param strategies: Strategies for generator node.
	:return: The best result dataframe.
	    It contains previous result columns and generator node's result columns.
	"""
	if not os.path.exists(node_line_dir):
		os.makedirs(node_line_dir)
	project_dir = pathlib.PurePath(node_line_dir).parent.parent
	node_dir = os.path.join(node_line_dir, "generator")  # node name
	if not os.path.exists(node_dir):
		os.makedirs(node_dir)
	qa_data = pd.read_parquet(
		os.path.join(project_dir, "data", "qa.parquet"), engine="pyarrow"
	)
	if "generation_gt" not in qa_data.columns:
		raise ValueError("You must have 'generation_gt' column in qa.parquet.")

	results, execution_times = zip(
		*map(
			lambda x: measure_speed(
				x[0].run_evaluator,
				project_dir=project_dir,
				previous_result=previous_result,
				**x[1],
			),
			zip(modules, module_params),
		)
	)
	average_times = list(map(lambda x: x / len(results[0]), execution_times))

	# get average token usage
	token_usages = list(map(lambda x: x["generated_tokens"].apply(len).mean(), results))

	# make rows to metric_inputs
	generation_gt = to_list(qa_data["generation_gt"].tolist())

	metric_inputs = [MetricInput(generation_gt=gen_gt) for gen_gt in generation_gt]

	metric_names, metric_params = cast_metrics(strategies.get("metrics"))
	if metric_names is None or len(metric_names) <= 0:
		raise ValueError("You must at least one metrics for generator evaluation.")
	results = list(
		map(
			lambda result: evaluate_generator_node(
				result, metric_inputs, strategies.get("metrics")
			),
			results,
		)
	)

	# save results to folder
	filepaths = list(
		map(lambda x: os.path.join(node_dir, f"{x}.parquet"), range(len(modules)))
	)
	list(
		map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths))
	)  # execute save to parquet
	filenames = list(map(lambda x: os.path.basename(x), filepaths))

	summary_df = pd.DataFrame(
		{
			"filename": filenames,
			"module_name": list(map(lambda module: module.__name__, modules)),
			"module_params": module_params,
			"execution_time": average_times,
			"average_output_token": token_usages,
			**{
				metric: list(map(lambda x: x[metric].mean(), results))
				for metric in metric_names
			},
		}
	)

	# filter by strategies
	if strategies.get("speed_threshold") is not None:
		results, filenames = filter_by_threshold(
			results, average_times, strategies["speed_threshold"], filenames
		)
	if strategies.get("token_threshold") is not None:
		results, filenames = filter_by_threshold(
			results, token_usages, strategies["token_threshold"], filenames
		)
	selected_result, selected_filename = select_best(
		results, metric_names, filenames, strategies.get("strategy", "mean")
	)
	best_result = pd.concat([previous_result, selected_result], axis=1)

	# add 'is_best' column at summary file
	summary_df["is_best"] = summary_df["filename"] == selected_filename

	# save files
	summary_df.to_csv(os.path.join(node_dir, "summary.csv"), index=False)
	best_result.to_parquet(
		os.path.join(
			node_dir, f"best_{os.path.splitext(selected_filename)[0]}.parquet"
		),
		index=False,
	)
	return best_result


def evaluate_generator_node(
	result_df: pd.DataFrame,
	metric_inputs: List[MetricInput],
	metrics: Union[List[str], List[Dict]],
):
	@evaluate_generation(metric_inputs=metric_inputs, metrics=metrics)
	def evaluate_generation_module(df: pd.DataFrame):
		return (
			df["generated_texts"].tolist(),
			df["generated_tokens"].tolist(),
			df["generated_log_probs"].tolist(),
		)

	return evaluate_generation_module(result_df)
