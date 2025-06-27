import abc
import logging
import os
from typing import List, Union, Tuple

import pandas as pd

from autorag.schema import BaseModule
from autorag.utils import validate_qa_dataset

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
