import os
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
import pandas as pd

from autorag.nodes.retrieval.base import HybridRetrieval
from autorag.utils.util import pop_params, fetch_contents, result_to_dataframe


def normalize_mm(scores: List[str], fixed_min_value: float = 0):
	arr = np.array(scores)
	max_value = np.max(arr)
	min_value = np.min(arr)
	norm_score = (arr - min_value) / (max_value - min_value)
	return norm_score


def normalize_tmm(scores: List[str], fixed_min_value: float):
	arr = np.array(scores)
	max_value = np.max(arr)
	norm_score = (arr - fixed_min_value) / (max_value - fixed_min_value)
	return norm_score


def normalize_z(scores: List[str], fixed_min_value: float = 0):
	arr = np.array(scores)
	mean_value = np.mean(arr)
	std_value = np.std(arr)
	norm_score = (arr - mean_value) / std_value
	return norm_score


def normalize_dbsf(scores: List[str], fixed_min_value: float = 0):
	arr = np.array(scores)
	mean_value = np.mean(arr)
	std_value = np.std(arr)
	min_value = mean_value - 3 * std_value
	max_value = mean_value + 3 * std_value
	norm_score = (arr - min_value) / (max_value - min_value)
	return norm_score


normalize_method_dict = {
	"mm": normalize_mm,
	"tmm": normalize_tmm,
	"z": normalize_z,
	"dbsf": normalize_dbsf,
}


class HybridCC(HybridRetrieval):
	def _pure(
		self,
		ids: Tuple,
		scores: Tuple,
		top_k: int,
		weight: float,
		normalize_method: str = "mm",
		semantic_theoretical_min_value: float = -1.0,
		lexical_theoretical_min_value: float = 0.0,
	):
		return hybrid_cc(
			ids,
			scores,
			top_k,
			weight,
			normalize_method,
			semantic_theoretical_min_value,
			lexical_theoretical_min_value,
		)

	@classmethod
	def run_evaluator(
		cls,
		project_dir: Union[str, Path],
		previous_result: pd.DataFrame,
		*args,
		**kwargs,
	):
		if "ids" in kwargs and "scores" in kwargs:
			data_dir = os.path.join(project_dir, "data")
			corpus_df = pd.read_parquet(
				os.path.join(data_dir, "corpus.parquet"), engine="pyarrow"
			)

			params = pop_params(hybrid_cc, kwargs)
			assert (
				"ids" in params and "scores" in params and "top_k" in params
			), "ids, scores, and top_k must be specified."

			@result_to_dataframe(
				["retrieved_contents", "retrieved_ids", "retrieve_scores"]
			)
			def __cc(**cc_params):
				ids, scores = hybrid_cc(**cc_params)
				contents = fetch_contents(corpus_df, ids)
				return contents, ids, scores

			return __cc(**params)
		else:
			assert (
				"target_modules" in kwargs and "target_module_params" in kwargs
			), "target_modules and target_module_params must be specified if there is not ids and scores."
			instance = cls(project_dir, *args, **kwargs)
			result = instance.pure(previous_result, *args, **kwargs)
			del instance
			return result


def hybrid_cc(
	ids: Tuple,
	scores: Tuple,
	top_k: int,
	weight: float,
	normalize_method: str = "mm",
	semantic_theoretical_min_value: float = -1.0,
	lexical_theoretical_min_value: float = 0.0,
) -> Tuple[List[List[str]], List[List[float]]]:
	"""
	Hybrid CC function.
	CC (convex combination) is a method to fuse lexical and semantic retrieval results.
	It is a method that first normalizes the scores of each retrieval result,
	and then combines them with the given weights.
	It is uniquer than other retrieval modules, because it does not really execute retrieval,
	but just fuse the results of other retrieval functions.
	So you have to run more than two retrieval modules before running this function.
	And collect ids and scores result from each retrieval module.
	Make it as tuple and input it to this function.

	:param ids: The tuple of ids that you want to fuse.
	    The length of this must be the same as the length of scores.
	    The semantic retrieval ids must be the first index.
	:param scores: The retrieve scores that you want to fuse.
	    The length of this must be the same as the length of ids.
	    The semantic retrieval scores must be the first index.
	:param top_k: The number of passages to be retrieved.
	:param normalize_method: The normalization method to use.
	  There are some normalization method that you can use at the hybrid cc method.
	  AutoRAG support following.
	    - `mm`: Min-max scaling
	    - `tmm`: Theoretical min-max scaling
	    - `z`: z-score normalization
	    - `dbsf`: 3-sigma normalization
	:param weight: The weight value. If the weight is 1.0, it means the
	  weight to the semantic module will be 1.0 and weight to the lexical module will be 0.0.
	:param semantic_theoretical_min_value: This value used by `tmm` normalization method. You can set the
	    theoretical minimum value by yourself. Default is -1.
	:param lexical_theoretical_min_value: This value used by `tmm` normalization method. You can set the
	    theoretical minimum value by yourself. Default is 0.
	:return: The tuple of ids and fused scores that fused by CC. Plus, the third element is selected weight value.
	"""
	assert len(ids) == len(scores), "The length of ids and scores must be the same."
	assert len(ids) > 1, "You must input more than one retrieval results."
	assert top_k > 0, "top_k must be greater than 0."
	assert weight >= 0, "The weight must be greater than 0."
	assert weight <= 1, "The weight must be less than 1."

	df = pd.DataFrame(
		{
			"semantic_ids": ids[0],
			"lexical_ids": ids[1],
			"semantic_score": scores[0],
			"lexical_score": scores[1],
		}
	)

	def cc_pure_apply(row):
		return fuse_per_query(
			row["semantic_ids"],
			row["lexical_ids"],
			row["semantic_score"],
			row["lexical_score"],
			normalize_method=normalize_method,
			weight=weight,
			top_k=top_k,
			semantic_theoretical_min_value=semantic_theoretical_min_value,
			lexical_theoretical_min_value=lexical_theoretical_min_value,
		)

	# fixed weight
	df[["cc_id", "cc_score"]] = df.apply(
		lambda row: cc_pure_apply(row), axis=1, result_type="expand"
	)
	return df["cc_id"].tolist(), df["cc_score"].tolist()


def fuse_per_query(
	semantic_ids: List[str],
	lexical_ids: List[str],
	semantic_scores: List[float],
	lexical_scores: List[float],
	normalize_method: str,
	weight: float,
	top_k: int,
	semantic_theoretical_min_value: float,
	lexical_theoretical_min_value: float,
):
	normalize_func = normalize_method_dict[normalize_method]
	norm_semantic_scores = normalize_func(
		semantic_scores, semantic_theoretical_min_value
	)
	norm_lexical_scores = normalize_func(lexical_scores, lexical_theoretical_min_value)
	ids = [semantic_ids, lexical_ids]
	scores = [norm_semantic_scores, norm_lexical_scores]
	df = pd.concat(
		[pd.Series(dict(zip(_id, score))) for _id, score in zip(ids, scores)], axis=1
	)
	df.columns = ["semantic", "lexical"]
	df = df.fillna(0)
	df["weighted_sum"] = df.mul((weight, 1.0 - weight)).sum(axis=1)
	df = df.sort_values(by="weighted_sum", ascending=False)
	return df.index.tolist()[:top_k], df["weighted_sum"][:top_k].tolist()
