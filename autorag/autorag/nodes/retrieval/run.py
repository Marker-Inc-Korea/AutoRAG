import logging
import os
import pathlib
from copy import deepcopy
from typing import List, Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd

from autorag.evaluation import evaluate_retrieval
from autorag.schema.metricinput import MetricInput
from autorag.strategy import measure_speed, filter_by_threshold, select_best
from autorag.support import get_support_modules
from autorag.utils.util import get_best_row, to_list, apply_recursive

logger = logging.getLogger("AutoRAG")

semantic_module_names = ["vectordb", "VectorDB"]
lexical_module_names = ["bm25", "BM25"]
hybrid_module_names = ["hybrid_rrf", "hybrid_cc", "HybridCC", "HybridRRF"]


def run_retrieval_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run evaluation and select the best module among retrieval node results.

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

	save_dir = os.path.join(node_line_dir, "retrieval")  # node name
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	def run(input_modules, input_module_params) -> Tuple[List[pd.DataFrame], List]:
		"""
		Run input modules and parameters.

		:param input_modules: Input modules
		:param input_module_params: Input module parameters
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

	def find_best(results, average_times, filenames):
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

	filename_first = 0
	# run semantic modules
	logger.info("Running retrieval node - semantic retrieval module...")
	if any([module.__name__ in semantic_module_names for module in modules]):
		semantic_modules, semantic_module_params = zip(
			*filter(
				lambda x: x[0].__name__ in semantic_module_names,
				zip(modules, module_params),
			)
		)
		semantic_results, semantic_times = run(semantic_modules, semantic_module_params)
		semantic_summary_df = save_and_summary(
			semantic_modules,
			semantic_module_params,
			semantic_results,
			semantic_times,
			filename_first,
		)
		semantic_selected_result, semantic_selected_filename = find_best(
			semantic_results, semantic_times, semantic_summary_df["filename"].tolist()
		)
		semantic_summary_df["is_best"] = (
			semantic_summary_df["filename"] == semantic_selected_filename
		)
		filename_first += len(semantic_modules)
	else:
		(
			semantic_selected_filename,
			semantic_summary_df,
			semantic_results,
			semantic_times,
		) = None, pd.DataFrame(), [], []
	# run lexical modules
	logger.info("Running retrieval node - lexical retrieval module...")
	if any([module.__name__ in lexical_module_names for module in modules]):
		lexical_modules, lexical_module_params = zip(
			*filter(
				lambda x: x[0].__name__ in lexical_module_names,
				zip(modules, module_params),
			)
		)
		lexical_results, lexical_times = run(lexical_modules, lexical_module_params)
		lexical_summary_df = save_and_summary(
			lexical_modules,
			lexical_module_params,
			lexical_results,
			lexical_times,
			filename_first,
		)
		lexical_selected_result, lexical_selected_filename = find_best(
			lexical_results, lexical_times, lexical_summary_df["filename"].tolist()
		)
		lexical_summary_df["is_best"] = (
			lexical_summary_df["filename"] == lexical_selected_filename
		)
		filename_first += len(lexical_modules)
	else:
		(
			lexical_selected_filename,
			lexical_summary_df,
			lexical_results,
			lexical_times,
		) = None, pd.DataFrame(), [], []

	logger.info("Running retrieval node - hybrid retrieval module...")
	# Next, run hybrid retrieval
	if any([module.__name__ in hybrid_module_names for module in modules]):
		hybrid_modules, hybrid_module_params = zip(
			*filter(
				lambda x: x[0].__name__ in hybrid_module_names,
				zip(modules, module_params),
			)
		)
		if all(
			["target_module_params" in x for x in hybrid_module_params]
		):  # for Runner.run
			# If target_module_params are already given, run hybrid retrieval directly
			hybrid_results, hybrid_times = run(hybrid_modules, hybrid_module_params)
			hybrid_summary_df = save_and_summary(
				hybrid_modules,
				hybrid_module_params,
				hybrid_results,
				hybrid_times,
				filename_first,
			)
			filename_first += len(hybrid_modules)
		else:  # for Evaluator
			# get id and score
			ids_scores = get_ids_and_scores(
				save_dir,
				[semantic_selected_filename, lexical_selected_filename],
				semantic_summary_df,
				lexical_summary_df,
				previous_result,
			)
			hybrid_module_params = list(
				map(lambda x: {**x, **ids_scores}, hybrid_module_params)
			)

			# optimize each modules
			real_hybrid_times = [
				get_hybrid_execution_times(semantic_summary_df, lexical_summary_df)
			] * len(hybrid_module_params)
			hybrid_times = real_hybrid_times.copy()
			hybrid_results = []
			for module, module_param in zip(hybrid_modules, hybrid_module_params):
				module_result_df, module_best_weight = optimize_hybrid(
					module,
					module_param,
					strategies,
					metric_inputs,
					project_dir,
					previous_result,
				)
				module_param["weight"] = module_best_weight
				hybrid_results.append(module_result_df)

			hybrid_summary_df = save_and_summary(
				hybrid_modules,
				hybrid_module_params,
				hybrid_results,
				hybrid_times,
				filename_first,
			)
			filename_first += len(hybrid_modules)
			hybrid_summary_df["execution_time"] = hybrid_times
			best_semantic_summary_row = semantic_summary_df.loc[
				semantic_summary_df["is_best"]
			].iloc[0]
			best_lexical_summary_row = lexical_summary_df.loc[
				lexical_summary_df["is_best"]
			].iloc[0]
			target_modules = (
				best_semantic_summary_row["module_name"],
				best_lexical_summary_row["module_name"],
			)
			target_module_params = (
				best_semantic_summary_row["module_params"],
				best_lexical_summary_row["module_params"],
			)
			hybrid_summary_df = edit_summary_df_params(
				hybrid_summary_df, target_modules, target_module_params
			)
	else:
		if any([module.__name__ in hybrid_module_names for module in modules]):
			logger.warning(
				"You must at least one semantic module and lexical module for hybrid evaluation."
				"Passing hybrid module."
			)
		_, hybrid_summary_df, hybrid_results, hybrid_times = (
			None,
			pd.DataFrame(),
			[],
			[],
		)

	summary = pd.concat(
		[semantic_summary_df, lexical_summary_df, hybrid_summary_df], ignore_index=True
	)
	results = semantic_results + lexical_results + hybrid_results
	average_times = semantic_times + lexical_times + hybrid_times
	filenames = summary["filename"].tolist()

	# filter by strategies
	selected_result, selected_filename = find_best(results, average_times, filenames)
	best_result = pd.concat([previous_result, selected_result], axis=1)

	# add summary.csv 'is_best' column
	summary["is_best"] = summary["filename"] == selected_filename

	# save the result files
	best_result.to_parquet(
		os.path.join(
			save_dir, f"best_{os.path.splitext(selected_filename)[0]}.parquet"
		),
		index=False,
	)
	summary.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
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


def edit_summary_df_params(
	summary_df: pd.DataFrame, target_modules, target_module_params
) -> pd.DataFrame:
	def delete_ids_scores(x):
		del x["ids"]
		del x["scores"]
		return x

	summary_df["module_params"] = summary_df["module_params"].apply(delete_ids_scores)
	summary_df["new_params"] = [
		{"target_modules": target_modules, "target_module_params": target_module_params}
	] * len(summary_df)
	summary_df["module_params"] = summary_df.apply(
		lambda row: {**row["module_params"], **row["new_params"]}, axis=1
	)
	summary_df = summary_df.drop(columns=["new_params"])
	return summary_df


def get_ids_and_scores(
	node_dir: str,
	filenames: List[str],
	semantic_summary_df: pd.DataFrame,
	lexical_summary_df: pd.DataFrame,
	previous_result,
) -> Dict[str, Tuple[List[List[str]], List[List[float]]]]:
	project_dir = pathlib.PurePath(node_dir).parent.parent.parent
	best_results_df = list(
		map(
			lambda filename: pd.read_parquet(
				os.path.join(node_dir, filename), engine="pyarrow"
			),
			filenames,
		)
	)
	ids = tuple(
		map(lambda df: df["retrieved_ids"].apply(list).tolist(), best_results_df)
	)
	scores = tuple(
		map(lambda df: df["retrieve_scores"].apply(list).tolist(), best_results_df)
	)
	# search non-duplicate ids
	semantic_ids = deepcopy(ids[0])
	lexical_ids = deepcopy(ids[1])

	def get_non_duplicate_ids(target_ids, compare_ids) -> List[List[str]]:
		"""
		Get non-duplicate ids from target_ids and compare_ids.
		If you want to non-duplicate ids of semantic_ids, you have to put it at target_ids.
		"""
		result_ids = []
		assert len(target_ids) == len(compare_ids)
		for target_id_list, compare_id_list in zip(target_ids, compare_ids):
			query_duplicated = list(set(compare_id_list) - set(target_id_list))
			duplicate_list = query_duplicated if len(query_duplicated) != 0 else []
			result_ids.append(duplicate_list)
		return result_ids

	lexical_target_ids = get_non_duplicate_ids(lexical_ids, semantic_ids)
	semantic_target_ids = get_non_duplicate_ids(semantic_ids, lexical_ids)

	new_id_tuple = (
		[a + b for a, b in zip(semantic_ids, semantic_target_ids)],
		[a + b for a, b in zip(lexical_ids, lexical_target_ids)],
	)

	# search non-duplicate ids' scores
	new_semantic_scores = get_scores_by_ids(
		semantic_target_ids, semantic_summary_df, project_dir, previous_result
	)
	new_lexical_scores = get_scores_by_ids(
		lexical_target_ids, lexical_summary_df, project_dir, previous_result
	)

	new_score_tuple = (
		[a + b for a, b in zip(scores[0], new_semantic_scores)],
		[a + b for a, b in zip(scores[1], new_lexical_scores)],
	)
	return {
		"ids": new_id_tuple,
		"scores": new_score_tuple,
	}


def get_scores_by_ids(
	ids: List[List[str]], module_summary_df: pd.DataFrame, project_dir, previous_result
) -> List[List[float]]:
	module_name = get_best_row(module_summary_df)["module_name"]
	module_params = get_best_row(module_summary_df)["module_params"]
	module = get_support_modules(module_name)
	result_df = module.run_evaluator(
		project_dir=project_dir,
		previous_result=previous_result,
		ids=ids,
		**module_params,
	)
	return to_list(result_df["retrieve_scores"].tolist())


def find_unique_elems(list1: List[str], list2: List[str]) -> List[str]:
	return list(set(list1).symmetric_difference(set(list2)))


def get_hybrid_execution_times(lexical_summary, semantic_summary) -> float:
	lexical_execution_time = lexical_summary.loc[lexical_summary["is_best"]].iloc[0][
		"execution_time"
	]
	semantic_execution_time = semantic_summary.loc[semantic_summary["is_best"]].iloc[0][
		"execution_time"
	]
	return lexical_execution_time + semantic_execution_time


def optimize_hybrid(
	hybrid_module_func: Callable,
	hybrid_module_param: Dict,
	strategy: Dict,
	input_metrics: List[MetricInput],
	project_dir,
	previous_result,
):
	if (
		hybrid_module_func.__name__ == "HybridRRF"
		or hybrid_module_func.__name__ == "hybrid_rrf"
	):
		weight_range = hybrid_module_param.pop("weight_range", (4, 80))
		test_weight_size = weight_range[1] - weight_range[0] + 1
	elif (
		hybrid_module_func.__name__ == "HybridCC"
		or hybrid_module_func.__name__ == "hybrid_cc"
	):
		weight_range = hybrid_module_param.pop("weight_range", (0.0, 1.0))
		test_weight_size = hybrid_module_param.pop("test_weight_size", 101)
	else:
		raise ValueError("You must input hybrid module function at hybrid_module_func.")

	weight_candidates = np.linspace(
		weight_range[0], weight_range[1], test_weight_size
	).tolist()

	result_list = []
	for weight_value in weight_candidates:
		result_df = hybrid_module_func.run_evaluator(
			project_dir=project_dir,
			previous_result=previous_result,
			weight=weight_value,
			**hybrid_module_param,
		)
		result_list.append(result_df)

		# evaluate here
	if strategy.get("metrics") is None:
		raise ValueError("You must at least one metrics for retrieval evaluation.")
	result_list = list(
		map(
			lambda x: evaluate_retrieval_node(
				x,
				input_metrics,
				strategy.get("metrics"),
			),
			result_list,
		)
	)

	# select best result
	best_result_df, best_weight = select_best(
		result_list,
		strategy.get("metrics"),
		metadatas=weight_candidates,
		strategy_name=strategy.get("strategy", "normalize_mean"),
	)
	return best_result_df, best_weight
