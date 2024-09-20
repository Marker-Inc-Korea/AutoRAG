import abc
import functools
import logging
import os
from pathlib import Path
from typing import List, Union, Tuple

import pandas as pd

from autorag.schema import BaseModule
from autorag.support import get_support_modules
from autorag.utils import fetch_contents, result_to_dataframe, validate_qa_dataset
from autorag.utils.util import pop_params

logger = logging.getLogger("AutoRAG")


class BaseRetrieval(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: str, *args, **kwargs):
		logger.info(f"Initialize retrieval node - {self.__class__.__name__}")

		self.resources_dir = os.path.join(project_dir, "resources")
		data_dir = os.path.join(project_dir, "data")
		# fetch data from corpus_data
		self.corpus_df = pd.read_parquet(
			os.path.join(data_dir, "corpus.parquet"), engine="pyarrow"
		)

	def __del__(self):
		logger.info(f"Deleting retrieval node - {self.__class__.__name__} module...")

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		logger.info(f"Running retrieval node - {self.__class__.__name__} module...")
		validate_qa_dataset(previous_result)
		# find queries columns & type cast queries
		assert (
			"query" in previous_result.columns
		), "previous_result must have query column."
		if "queries" not in previous_result.columns:
			previous_result["queries"] = previous_result["query"]
		previous_result.loc[:, "queries"] = previous_result["queries"].apply(
			cast_queries
		)
		queries = previous_result["queries"].tolist()
		return queries


class HybridRetrieval(BaseRetrieval, metaclass=abc.ABCMeta):
	def __init__(
		self, project_dir: str, target_modules, target_module_params, *args, **kwargs
	):
		super().__init__(project_dir)
		self.target_modules = list(
			map(
				lambda x, y: get_support_modules(x)(
					**y,
					project_dir=project_dir,
				),
				target_modules,
				target_module_params,
			)
		)
		self.target_module_params = target_module_params

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		result_dfs: List[pd.DataFrame] = list(
			map(
				lambda x, y: x.pure(
					**y,
					previous_result=previous_result,
				),
				self.target_modules,
				self.target_module_params,
			)
		)
		ids = tuple(
			map(lambda df: df["retrieved_ids"].apply(list).tolist(), result_dfs)
		)
		scores = tuple(
			map(
				lambda df: df["retrieve_scores"].apply(list).tolist(),
				result_dfs,
			)
		)

		_pure_params = pop_params(self._pure, kwargs)
		if "ids" in _pure_params or "scores" in _pure_params:
			raise ValueError(
				"With specifying ids or scores, you must use HybridRRF.run_evaluator instead."
			)
		ids, scores = self._pure(ids=ids, scores=scores, **_pure_params)
		contents = fetch_contents(self.corpus_df, ids)
		return contents, ids, scores


def retrieval_node(func):
	"""
	Load resources for running retrieval_node.
	For example, it loads bm25 corpus for bm25 retrieval.

	:param func: Retrieval function that returns a list of ids and a list of scores
	:return: A pandas Dataframe that contains retrieved contents, retrieved ids, and retrieve scores.
	    The column name will be "retrieved_contents", "retrieved_ids", and "retrieve_scores".
	"""

	@functools.wraps(func)
	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def wrapper(
		project_dir: Union[str, Path], previous_result: pd.DataFrame, **kwargs
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		validate_qa_dataset(previous_result)
		resources_dir = os.path.join(project_dir, "resources")
		data_dir = os.path.join(project_dir, "data")

		if func.__name__ == "bm25":
			# check if bm25_path and file exist
			bm25_tokenizer = kwargs.get("bm25_tokenizer", None)
			if bm25_tokenizer is None:
				bm25_tokenizer = "porter_stemmer"
			bm25_path = os.path.join(resources_dir, get_bm25_pkl_name(bm25_tokenizer))
			assert (
				bm25_path is not None
			), "bm25_path must be specified for using bm25 retrieval."
			assert os.path.exists(
				bm25_path
			), f"bm25_path {bm25_path} does not exist. Please ingest first."

		# find queries columns & type cast queries
		assert (
			"query" in previous_result.columns
		), "previous_result must have query column."
		if "queries" not in previous_result.columns:
			previous_result["queries"] = previous_result["query"]
		previous_result.loc[:, "queries"] = previous_result["queries"].apply(
			cast_queries
		)
		queries = previous_result["queries"].tolist()

		# run retrieval function
		if func.__name__ == "bm25":
			bm25_corpus = load_bm25_corpus(bm25_path)
			ids, scores = func(queries=queries, bm25_corpus=bm25_corpus, **kwargs)
		elif func.__name__ == "vectordb":
			ids, scores = func(
				queries=queries,
				collection=chroma_collection,
				embedding_model=embedding_model,
				**kwargs,
			)

		elif func.__name__ in ["hybrid_rrf", "hybrid_cc"]:
			if "ids" in kwargs and "scores" in kwargs:  # ordinary run_evaluate
				ids, scores = func(**kwargs)
			else:  # => for Runner.run
				# TODO: At hybrid retrieval, you can "override run_evaluator"... Wow!!! So Magical Jax Structure so sexy an hot
				if not (
					"target_modules" in kwargs and "target_module_params" in kwargs
				):
					raise ValueError(
						f"If there are no ids and scores specified, target_modules and target_module_params must be specified for using {func.__name__}."
					)
				target_modules = kwargs.pop("target_modules")
				target_module_params = kwargs.pop("target_module_params")
				result_dfs = list(
					map(
						lambda x: get_support_modules(x[0])(
							**x[1],
							project_dir=project_dir,
							previous_result=previous_result,
						),
						zip(target_modules, target_module_params),
					)
				)
				ids = tuple(
					map(lambda df: df["retrieved_ids"].apply(list).tolist(), result_dfs)
				)
				scores = tuple(
					map(
						lambda df: df["retrieve_scores"].apply(list).tolist(),
						result_dfs,
					)
				)
				ids, scores = func(ids=ids, scores=scores, **kwargs)
		else:
			raise ValueError("invalid func name for using retrieval_io decorator.")

		# fetch data from corpus_data
		corpus_data = pd.read_parquet(
			os.path.join(data_dir, "corpus.parquet"), engine="pyarrow"
		)
		contents = fetch_contents(corpus_data, ids)

		return contents, ids, scores

	return wrapper


def cast_queries(queries: Union[str, List[str]]) -> List[str]:
	if isinstance(queries, str):
		return [queries]
	elif isinstance(queries, List):
		return queries
	else:
		raise ValueError(f"queries must be str or list, but got {type(queries)}")


def evenly_distribute_passages(
	ids: List[List[str]], scores: List[List[float]], top_k: int
) -> Tuple[List[str], List[float]]:
	assert len(ids) == len(scores), "ids and scores must have same length."
	query_cnt = len(ids)
	avg_len = top_k // query_cnt
	remainder = top_k % query_cnt

	new_ids = []
	new_scores = []
	for i in range(query_cnt):
		if i < remainder:
			new_ids.extend(ids[i][: avg_len + 1])
			new_scores.extend(scores[i][: avg_len + 1])
		else:
			new_ids.extend(ids[i][:avg_len])
			new_scores.extend(scores[i][:avg_len])

	return new_ids, new_scores


def get_bm25_pkl_name(bm25_tokenizer: str):
	bm25_tokenizer = bm25_tokenizer.replace("/", "")
	return f"bm25_{bm25_tokenizer}.pkl"
