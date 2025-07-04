import os
import pathlib
from typing import List, Dict, Union

import pandas as pd

from autorag.evaluation import evaluate_retrieval
from autorag.evaluation.retrieval import RETRIEVAL_METRIC_FUNC_DICT
from autorag.nodes.retrieval.run_util import save_and_summary, find_best
from autorag.schema.metricinput import MetricInput
from autorag.strategy import measure_speed
from autorag.utils.util import apply_recursive, to_list


def run_hybrid_retrieval_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run the hybrid retrieval node with the given modules and parameters.

	:param modules: List of hybrid retrieval modules to run.
	:param module_params: List of parameters for each module.
	:param previous_result: DataFrame containing the previous results.
	:param node_line_dir: Directory where the results will be saved.
	:param strategies: Dictionary containing strategies for filtering and evaluation.
	:return: DataFrame containing the results of the hybrid retrieval.
	"""
	if not os.path.exists(node_line_dir):
		os.makedirs(node_line_dir)
	project_dir = pathlib.PurePath(node_line_dir).parent.parent
	qa_df = pd.read_parquet(
		os.path.join(project_dir, "data", "qa.parquet"), engine="pyarrow"
	)
	retrieval_gt = qa_df["retrieval_gt"].tolist()
	retrieval_gt = apply_recursive(lambda x: str(x), to_list(retrieval_gt))

	# make rows to metric_inputs
	metric_inputs = [
		MetricInput(retrieval_gt=ret_gt, query=query, generation_gt=gen_gt)
		for ret_gt, query, gen_gt in zip(
			retrieval_gt, qa_df["query"].tolist(), qa_df["generation_gt"].tolist()
		)
	]

	save_dir = os.path.join(node_line_dir, "hybrid_retrieval")  # node name
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	results, execution_times = [], []
	for module, module_param in zip(modules, module_params):
		result, speed = measure_speed(
			module.run_evaluator,
			project_dir=project_dir,
			previous_result=previous_result,
			strategy=strategies,
			input_metrics=metric_inputs,
			**module_param,
		)
		if "best_weight" in result.keys():
			module_param["weight"] = result["best_weight"]
		results.append(result["best_result"])
		execution_times.append(speed)

	hybrid_times = list(map(lambda x: x / len(results[0]), execution_times))

	# run metrics before filtering
	if strategies.get("metrics") is None:
		raise ValueError("You must at least one metrics for retrieval evaluation.")
	results = list(
		map(
			lambda x: evaluate_retrieval_node(
				x,
				metric_inputs,
				strategies.get("metrics"),
			),
			results,
		)
	)

	summary_df = save_and_summary(
		modules,
		module_params,
		results,
		hybrid_times,
		0,
		save_dir,
		strategies,
	)
	selected_result, selected_filename = find_best(
		results, hybrid_times, summary_df["filename"].tolist(), strategies
	)
	summary_df["is_best"] = summary_df["filename"] == selected_filename
	previous_result.drop(
		columns=list(RETRIEVAL_METRIC_FUNC_DICT.keys()), inplace=True, errors="ignore"
	)
	best_result = pd.concat([previous_result, selected_result], axis=1)
	best_result.to_parquet(
		os.path.join(
			save_dir, f"best_{os.path.splitext(selected_filename)[0]}.parquet"
		),
		index=False,
	)
	summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
	return best_result


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
