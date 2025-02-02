import os.path
import pathlib
from typing import List, Dict

import pandas as pd

from autorag.evaluation.metric import (
	retrieval_token_recall,
	retrieval_token_precision,
	retrieval_token_f1,
)
from autorag.schema.metricinput import MetricInput
from autorag.strategy import measure_speed, filter_by_threshold, select_best
from autorag.utils.util import fetch_contents


def run_passage_compressor_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run evaluation and select the best module among passage compressor modules.

	:param modules: Passage compressor modules to run.
	:param module_params: Passage compressor module parameters.
	:param previous_result: Previous result dataframe.
	    Could be retrieval, reranker modules result.
	    It means it must contain 'query', 'retrieved_contents', 'retrieved_ids', 'retrieve_scores' columns.
	:param node_line_dir: This node line's directory.
	:param strategies: Strategies for passage compressor node.
	    In this node, we use
	    You can skip evaluation when you use only one module and a module parameter.
	:return: The best result dataframe with previous result columns.
	    This node will replace 'retrieved_contents' to compressed passages, so its length will be one.
	"""
	if not os.path.exists(node_line_dir):
		os.makedirs(node_line_dir)
	project_dir = pathlib.PurePath(node_line_dir).parent.parent
	data_dir = os.path.join(project_dir, "data")
	save_dir = os.path.join(node_line_dir, "passage_compressor")
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# make retrieval contents gt
	qa_data = pd.read_parquet(os.path.join(data_dir, "qa.parquet"), engine="pyarrow")
	corpus_data = pd.read_parquet(
		os.path.join(data_dir, "corpus.parquet"), engine="pyarrow"
	)
	# check qa_data have retrieval_gt
	assert all(
		len(x[0]) > 0 for x in qa_data["retrieval_gt"].tolist()
	), "Can't use passage compressor if you don't have retrieval gt values in QA dataset."

	# run modules
	results, execution_times = zip(
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
	results = list(results)
	average_times = list(map(lambda x: x / len(results[0]), execution_times))

	retrieval_gt_contents = list(
		map(lambda x: fetch_contents(corpus_data, x), qa_data["retrieval_gt"].tolist())
	)

	metric_inputs = [
		MetricInput(retrieval_gt_contents=ret_cont_gt)
		for ret_cont_gt in retrieval_gt_contents
	]

	# run metrics before filtering
	if strategies.get("metrics") is None:
		raise ValueError(
			"You must at least one metrics for retrieval contents evaluation."
			"It can be 'retrieval_token_f1', 'retrieval_token_precision', 'retrieval_token_recall'."
		)
	results = list(
		map(
			lambda x: evaluate_passage_compressor_node(
				x, metric_inputs, strategies.get("metrics")
			),
			results,
		)
	)

	# save results to folder
	filepaths = list(
		map(lambda x: os.path.join(save_dir, f"{x}.parquet"), range(len(modules)))
	)
	list(
		map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths))
	)  # execute save to parquet
	filenames = list(map(lambda x: os.path.basename(x), filepaths))

	# make summary file
	summary_df = pd.DataFrame(
		{
			"filename": filenames,
			"module_name": list(map(lambda module: module.__name__, modules)),
			"module_params": module_params,
			"execution_time": average_times,
			**{
				f"passage_compressor_{metric}": list(
					map(lambda result: result[metric].mean(), results)
				)
				for metric in strategies.get("metrics")
			},
		}
	)

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
	new_retrieved_contents = selected_result["retrieved_contents"]
	previous_result["retrieved_contents"] = new_retrieved_contents
	selected_result = selected_result.drop(columns=["retrieved_contents"])
	best_result = pd.concat([previous_result, selected_result], axis=1)

	# add 'is_best' column to summary file
	summary_df["is_best"] = summary_df["filename"] == selected_filename

	# add prefix 'passage_compressor' to best_result columns
	best_result = best_result.rename(
		columns={
			metric_name: f"passage_compressor_{metric_name}"
			for metric_name in strategies.get("metrics")
		}
	)

	# save the result files
	best_result.to_parquet(
		os.path.join(
			save_dir, f"best_{os.path.splitext(selected_filename)[0]}.parquet"
		),
		index=False,
	)
	summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
	return best_result


def evaluate_passage_compressor_node(
	result_df: pd.DataFrame, metric_inputs: List[MetricInput], metrics: List[str]
):
	metric_funcs = {
		retrieval_token_recall.__name__: retrieval_token_recall,
		retrieval_token_precision.__name__: retrieval_token_precision,
		retrieval_token_f1.__name__: retrieval_token_f1,
	}
	for metric_input, generated_text in zip(
		metric_inputs, result_df["retrieved_contents"].tolist()
	):
		metric_input.retrieved_contents = generated_text
	metrics = list(filter(lambda x: x in metric_funcs.keys(), metrics))
	if len(metrics) <= 0:
		raise ValueError(f"metrics must be one of {metric_funcs.keys()}")
	metrics_scores = dict(
		map(
			lambda metric: (
				metric,
				metric_funcs[metric](
					metric_inputs=metric_inputs,
				),
			),
			metrics,
		)
	)
	result_df = pd.concat([result_df, pd.DataFrame(metrics_scores)], axis=1)
	return result_df
