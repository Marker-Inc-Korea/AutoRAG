import functools
from typing import List

import numpy as np

from autorag.schema.metricinput import MetricInput
from autorag.utils.util import convert_inputs_to_list


def calculate_cosine_similarity(a, b):
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	similarity = dot_product / (norm_a * norm_b)
	return similarity


def calculate_l2_distance(a, b):
	return np.linalg.norm(a - b)


def calculate_inner_product(a, b):
	return np.dot(a, b)


def autorag_metric(fields_to_check: List[str]):
	def decorator_autorag_metric(func):
		@functools.wraps(func)
		@convert_inputs_to_list
		def wrapper(metric_inputs: List[MetricInput], **kwargs) -> List[float]:
			"""
			Use 'for loop' to run each metric input.
			Put the single metric input into the metric function.

			:param metric_inputs: A list MetricInput schema for AutoRAG metric.
			:param kwargs: The additional arguments for metric function.
			:return: A list of computed metric scores.
			"""
			results = []
			for metric_input in metric_inputs:
				if metric_input.is_fields_notnone(fields_to_check=fields_to_check):
					results.append(func(metric_input, **kwargs))
				else:
					results.append(None)
			return results

		return wrapper

	return decorator_autorag_metric


def autorag_metric_loop(fields_to_check: List[str]):
	def decorator_autorag_generation_metric(func):
		@functools.wraps(func)
		@convert_inputs_to_list
		def wrapper(metric_inputs: List[MetricInput], **kwargs) -> List[float]:
			"""
			Put the list of metric inputs into the metric function.

			:param metric_inputs: A list MetricInput schema for AutoRAG metric.
			:param kwargs: The additional arguments for metric function.
			:return: A list of computed metric scores.
			"""
			bool_list = [
				metric_input.is_fields_notnone(fields_to_check=fields_to_check)
				for metric_input in metric_inputs
			]
			valid_inputs = [
				metric_input
				for metric_input, is_valid in zip(metric_inputs, bool_list)
				if is_valid
			]

			results = [None] * len(metric_inputs)
			if valid_inputs:
				processed_valid = func(valid_inputs, **kwargs)

				valid_index = 0
				for i, is_valid in enumerate(bool_list):
					if is_valid:
						results[i] = processed_valid[valid_index]
						valid_index += 1

			return results

		return wrapper

	return decorator_autorag_generation_metric
