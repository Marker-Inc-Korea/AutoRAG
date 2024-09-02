import functools
import warnings
from typing import List, Callable, Any, Tuple, Union, Dict

import pandas as pd

from autorag.evaluation.metric import (
	retrieval_recall,
	retrieval_precision,
	retrieval_f1,
	retrieval_ndcg,
	retrieval_mrr,
	retrieval_map,
)
from autorag.evaluation.util import cast_metrics
from autorag.schema.metricinput import MetricInput, METRIC_INPUT_DICT

RETRIEVAL_METRIC_FUNC_DICT = {
	func.__name__: func
	for func in [
		retrieval_recall,
		retrieval_precision,
		retrieval_f1,
		retrieval_ndcg,
		retrieval_mrr,
		retrieval_map,
	]
}
RETRIEVAL_NO_GT_METRIC_FUNC_DICT = {func.__name__: func for func in []}


def evaluate_retrieval(
		metric_inputs: List[MetricInput],
	metrics: Union[List[str], List[Dict]],
):
	def decorator_evaluate_retrieval(
		func: Callable[
			[Any], Tuple[List[List[str]], List[List[str]], List[List[float]]]
		],
	):
		"""
		Decorator for evaluating retrieval results.
		You can use this decorator to any method that returns (contents, scores, ids),
		which is the output of conventional retrieval modules.

		:param func: Must return (contents, scores, ids)
		:return: wrapper function that returns pd.DataFrame, which is the evaluation result.
		"""

		@functools.wraps(func)
		def wrapper(*args, **kwargs) -> pd.DataFrame:
			contents, pred_ids, scores = func(*args, **kwargs)

			metric_scores = {}
			metric_names, metric_params = cast_metrics(metrics)

			for metric_name, metric_param in zip(metric_names, metric_params):
				# Extract each required field from all payloads
				extracted_inputs = {field: [getattr(payload, field) for payload in metric_inputs] for field in
									METRIC_INPUT_DICT.get(metric_name, [])}

				if metric_name in RETRIEVAL_METRIC_FUNC_DICT:
					metric_func = RETRIEVAL_METRIC_FUNC_DICT[metric_name]
					metric_scores[metric_name] = metric_func(
						**extracted_inputs, pred_ids=pred_ids, **metric_param
					)
				elif metric_name in RETRIEVAL_NO_GT_METRIC_FUNC_DICT:
					metric_func = RETRIEVAL_NO_GT_METRIC_FUNC_DICT[metric_name]
					if 'queries' not in extracted_inputs.keys():
						raise ValueError(
							f"To using {metric_name}, you have to provide queries."
						)
					if 'generation_gt' not in extracted_inputs.keys():
						raise ValueError(
							f"To using {metric_name}, you have to provide generation ground truth."
						)
					metric_scores[metric_name] = metric_func(
						**extracted_inputs,
						retrieved_contents=contents,
						**metric_param,
					)
				else:
					warnings.warn(
						f"metric {metric_name} is not in supported metrics: {RETRIEVAL_METRIC_FUNC_DICT.keys()}"
						f" and {RETRIEVAL_NO_GT_METRIC_FUNC_DICT.keys()}"
						f"{metric_name} will be ignored."
					)

			metric_result_df = pd.DataFrame(metric_scores)
			execution_result_df = pd.DataFrame(
				{
					"retrieved_contents": contents,
					"retrieved_ids": pred_ids,
					"retrieve_scores": scores,
				}
			)
			result_df = pd.concat([execution_result_df, metric_result_df], axis=1)
			return result_df

		return wrapper

	return decorator_evaluate_retrieval
