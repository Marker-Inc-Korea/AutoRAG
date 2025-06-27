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


def run_lexical_retrieval_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run the semantic retrieval node.

	:param modules: Retrieval modules to run.
	:param module_params: Retrieval module parameters.
	:param previous_result: Previous result dataframe.
	    Could be query expansion's best result or qa data.
	:param node_line_dir: This node line's directory.
	:param strategies: Strategies for retrieval node.
	:return: The best result dataframe.
	    It contains previous result columns and retrieval node's result columns.
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

	save_dir = os.path.join(node_line_dir, "lexical_retrieval")
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Run the modules
	lexical_results, execution_times = zip(
		*map(
			lambda task: measure_speed(
				task[0].run_evaluator,
				project_dir=project_dir,
				previous_result=previous_result,
				**task[1],
			),
			zip(modules, module_params),
		)
	)
	lexical_times = list(map(lambda x: x / len(lexical_results[0]), execution_times))

	# run metrics
	if strategies.get("metrics") is None:
		raise ValueError("You must at least one metrics for retrieval evaluation.")
	lexical_results = list(
		map(
			lambda x: evaluate_lexical_retrieval_node(
				x,
				metric_inputs,
				strategies.get("metrics"),
			),
			lexical_results,
		)
	)

	lexical_summary_df = save_and_summary(
		modules,
		module_params,
		lexical_results,
		lexical_times,
		0,
		save_dir,
		strategies,
	)
	lexical_selected_result, lexical_selected_filename = find_best(
		lexical_results,
		lexical_times,
		lexical_summary_df["filename"].tolist(),
		strategies,
	)
	lexical_summary_df["is_best"] = (
		lexical_summary_df["filename"] == lexical_selected_filename
	)
	previous_result.drop(
		columns=list(RETRIEVAL_METRIC_FUNC_DICT.keys()), inplace=True, errors="ignore"
	)
	lexical_selected_result.rename(
		columns={
			"retrieved_contents": "retrieved_contents_lexical",
			"retrieved_ids": "retrieved_ids_lexical",
			"retrieve_scores": "retrieve_scores_lexical",
		},
		inplace=True,
	)
	best_result = pd.concat([previous_result, lexical_selected_result], axis=1)
	best_result.to_parquet(
		os.path.join(
			save_dir, f"best_{os.path.splitext(lexical_selected_filename)[0]}.parquet"
		),
		index=False,
	)
	lexical_summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
	return best_result


def evaluate_lexical_retrieval_node(
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
	    The columns will be 'retrieved_contents_lexical', 'retrieved_ids_lexical', 'retrieve_scores_lexical', and metric names.
	"""

	@evaluate_retrieval(
		metric_inputs=metric_inputs,
		metrics=metrics,
	)
	def evaluate_this_module(df: pd.DataFrame):
		return (
			df["retrieved_contents_lexical"].tolist(),
			df["retrieved_ids_lexical"].tolist(),
			df["retrieve_scores_lexical"].tolist(),
		)

	return evaluate_this_module(result_df)
