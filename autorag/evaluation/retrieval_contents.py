import functools
from typing import List, Callable, Any, Tuple

import pandas as pd

from autorag.evaluation.metric import (
	retrieval_token_f1,
	retrieval_token_precision,
	retrieval_token_recall,
)
from autorag.schema.payload import Payload, METRIC_INPUT_DICT


def evaluate_retrieval_contents(payloads: List[Payload], metrics: List[str]):
	def decorator_evaluate_retrieval_contents(
		func: Callable[
			[Any], Tuple[List[List[str]], List[List[str]], List[List[float]]]
		],
	):
		"""
		Decorator for evaluating retrieval contents.
		You can use this decorator to any method that returns (contents, scores, ids),
		which is the output of conventional retrieval modules.

		:param func: Must return (contents, scores, ids)
		:return: pd.DataFrame, which is the evaluation result and function result.
		"""

		@functools.wraps(func)
		def wrapper(*args, **kwargs) -> pd.DataFrame:
			contents, pred_ids, scores = func(*args, **kwargs)
			metric_funcs = {
				retrieval_token_recall.__name__: retrieval_token_recall,
				retrieval_token_precision.__name__: retrieval_token_precision,
				retrieval_token_f1.__name__: retrieval_token_f1,
			}

			metrics_scores = {}
			for metric in metrics:
				if metric not in metric_funcs:
					raise ValueError(
						f"metric {metric} is not in supported metrics: {metric_funcs.keys()}"
					)
				else:
					metric_func = metric_funcs[metric]
					# Extract each required field from all payloads
					extracted_inputs = {field: [getattr(payload, field) for payload in payloads] for field in
										METRIC_INPUT_DICT.get(metric_func.__name__, [])}
					metric_scores = metric_func(
						**extracted_inputs, pred_contents=contents
					)
					metrics_scores[metric] = metric_scores

			metric_result_df = pd.DataFrame(metrics_scores)
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

	return decorator_evaluate_retrieval_contents
