import functools
import warnings
from typing import List, Callable, Union, Dict

import pandas as pd

from autorag.evaluation.metric.generation import (
	bleu,
	meteor,
	rouge,
	sem_score,
	g_eval,
	bert_score,
	deepeval_faithfulness,
)
from autorag.evaluation.util import cast_metrics
from autorag.schema.metricinput import MetricInput

GENERATION_METRIC_FUNC_DICT = {
	func.__name__: func
	for func in [
		bleu,
		meteor,
		rouge,
		sem_score,
		g_eval,
		bert_score,
		deepeval_faithfulness,
	]
}


def evaluate_generation(
	metric_inputs: List[MetricInput], metrics: Union[List[str], List[Dict]]
):
	def decorator_evaluate_generation(func: Callable):
		@functools.wraps(func)
		def wrapper(*args, **kwargs) -> pd.DataFrame:
			generation_result = func(*args, **kwargs)
			if type(generation_result) is tuple:
				assert (
					type(generation_result[0]) is list
					and type(generation_result[0][0]) is str
				), "Input func must return string list as generated answer at the first return value."
				generated_str = generation_result[0]
			elif type(generation_result) is list:
				assert (
					type(generation_result[0]) is str
				), "Input func must return string list as generated answer at the first return value."
				generated_str = generation_result
			else:
				raise ValueError(
					"Input func must return string list as generated answer at the first return value."
				)
			for metric_input, generated_text in zip(metric_inputs, generated_str):
				metric_input.generated_texts = generated_text

			metric_scores = {}
			metric_names, metric_params = cast_metrics(metrics)

			for metric_name, metric_param in zip(metric_names, metric_params):
				if metric_name not in GENERATION_METRIC_FUNC_DICT:
					warnings.warn(
						f"metric {metric_name} is not in supported metrics: {GENERATION_METRIC_FUNC_DICT.keys()}"
						f"{metric_name} will be ignored."
					)
				else:
					metric_scores[metric_name] = GENERATION_METRIC_FUNC_DICT[
						metric_name
					](
						metric_inputs=metric_inputs,
						**metric_param,
					)

			metric_result_df = pd.DataFrame(metric_scores)
			execution_result_df = pd.DataFrame({"generated_texts": generated_str})
			if type(generation_result) is tuple:
				execution_result_df["generated_tokens"] = generation_result[1]
				execution_result_df["generated_log_probs"] = generation_result[2]

			result_df = pd.concat([execution_result_df, metric_result_df], axis=1)
			return result_df

		return wrapper

	return decorator_evaluate_generation
