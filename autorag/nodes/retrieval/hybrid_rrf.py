import os
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

from autorag.nodes.retrieval.base import HybridRetrieval
from autorag.utils.util import pop_params, fetch_contents, result_to_dataframe


class HybridRRF(HybridRetrieval):
	def _pure(self, ids, scores, top_k: int, weight: int = 60, rrf_k: int = -1):
		return hybrid_rrf(ids, scores, top_k, weight, rrf_k)

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

			params = pop_params(hybrid_rrf, kwargs)
			assert (
				"ids" in params and "scores" in params and "top_k" in params
			), "ids, scores, and top_k must be specified."

			@result_to_dataframe(
				["retrieved_contents", "retrieved_ids", "retrieve_scores"]
			)
			def __rrf(**rrf_params):
				ids, scores = hybrid_rrf(**rrf_params)
				contents = fetch_contents(corpus_df, ids)
				return contents, ids, scores

			return __rrf(**params)
		else:
			assert (
				"target_modules" in kwargs and "target_module_params" in kwargs
			), "target_modules and target_module_params must be specified if there is not ids and scores."
			instance = cls(project_dir, *args, **kwargs)
			result = instance.pure(previous_result, *args, **kwargs)
			del instance
			return result


def hybrid_rrf(
	ids: Tuple,
	scores: Tuple,
	top_k: int,
	weight: int = 60,
	rrf_k: int = -1,
) -> Tuple[List[List[str]], List[List[float]]]:
	"""
	Hybrid RRF function.
	RRF (Rank Reciprocal Fusion) is a method to fuse multiple retrieval results.
	It is common to fuse dense retrieval and sparse retrieval results using RRF.
	To use this function, you must input ids and scores as tuple.
	It is more unique than other retrieval modules because it does not really execute retrieval but just fuses
	the results of other retrieval functions.
	So you have to run more than two retrieval modules before running this function.
	And collect ids and scores result from each retrieval module.
	Make it as a tuple and input it to this function.

	:param ids: The tuple of ids that you want to fuse.
	    The length of this must be the same as the length of scores.
	:param scores: The retrieve scores that you want to fuse.
	    The length of this must be the same as the length of ids.
	:param top_k: The number of passages to be retrieved.
	:param weight: Hyperparameter for RRF.
	    It was originally rrf_k value.
	    Default is 60.
	    For more information, please visit our documentation.
	:param rrf_k: (Deprecated) Hyperparameter for RRF.
	    It was originally rrf_k value. Will remove at a further version.
	:return: The tuple of ids and fused scores that are fused by RRF.
	"""
	assert len(ids) == len(scores), "The length of ids and scores must be the same."
	assert len(ids) > 1, "You must input more than one retrieval results."
	assert top_k > 0, "top_k must be greater than 0."
	assert weight > 0, "rrf_k must be greater than 0."

	if rrf_k != -1:
		weight = int(rrf_k)
	else:
		weight = int(weight)

	id_df = pd.DataFrame({f"id_{i}": id_list for i, id_list in enumerate(ids)})
	score_df = pd.DataFrame(
		{f"score_{i}": score_list for i, score_list in enumerate(scores)}
	)
	df = pd.concat([id_df, score_df], axis=1)

	def rrf_pure_apply(row):
		ids_tuple = tuple(row[[f"id_{i}" for i in range(len(ids))]].values)
		scores_tuple = tuple(row[[f"score_{i}" for i in range(len(scores))]].values)
		return pd.Series(rrf_pure(ids_tuple, scores_tuple, weight, top_k))

	df[["rrf_id", "rrf_score"]] = df.apply(rrf_pure_apply, axis=1)
	return df["rrf_id"].tolist(), df["rrf_score"].tolist()


def rrf_pure(
	ids: Tuple, scores: Tuple, rrf_k: int, top_k: int
) -> Tuple[List[str], List[float]]:
	df = pd.concat(
		[pd.Series(dict(zip(_id, score))) for _id, score in zip(ids, scores)], axis=1
	)
	rank_df = df.rank(ascending=False, method="min")
	rank_df = rank_df.fillna(0)
	rank_df["rrf"] = rank_df.apply(lambda row: rrf_calculate(row, rrf_k), axis=1)
	rank_df = rank_df.sort_values(by="rrf", ascending=False)
	return rank_df.index.tolist()[:top_k], rank_df["rrf"].tolist()[:top_k]


def rrf_calculate(row, rrf_k):
	result = 0
	for r in row:
		if r == 0:
			continue
		result += 1 / (r + rrf_k)
	return result
