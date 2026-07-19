import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils import result_to_dataframe, fetch_contents


class TimeReranker(BasePassageReranker):
	def __init__(self, project_dir: str, *args, **kwargs):
		super().__init__(project_dir, *args, **kwargs)
		self.corpus_df = pd.read_parquet(
			os.path.join(project_dir, "data", "corpus.parquet"), engine="pyarrow"
		)

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		_, contents, scores, ids = self.cast_to_run(previous_result)
		metadatas = fetch_contents(self.corpus_df, ids, column_name="metadata")
		times = [
			[time["last_modified_datetime"] for time in time_list]
			for time_list in metadatas
		]
		top_k = kwargs.pop("top_k")
		return self._pure(contents, scores, ids, top_k, times)

	def _pure(
		self,
		contents_list: List[List[str]],
		scores_list: List[List[float]],
		ids_list: List[List[str]],
		top_k: int,
		time_list: List[List[datetime]],
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank the passages based on merely the datetime of the passage.
		It uses 'last_modified_datetime' key in the corpus metadata,
		so the metadata should be in the format of {'last_modified_datetime': datetime.datetime} at the corpus data file.

		:param contents_list: The list of lists of contents
		:param scores_list: The list of lists of scores from the initial ranking
		:param ids_list: The list of lists of ids
		:param top_k: The number of passages to be retrieved after reranking
		:param time_list: The metadata list of lists of datetime.datetime
			It automatically extracts the 'last_modified_datetime' key from the metadata in the corpus data.
		:return: The reranked contents, ids, and scores
		"""

		def sort_row(contents, scores, ids, time, top_k):
			combined = list(zip(contents, scores, ids, time))
			combined.sort(key=lambda x: x[3], reverse=True)
			sorted_contents, sorted_scores, sorted_ids, _ = zip(*combined)
			return (
				list(sorted_contents)[:top_k],
				list(sorted_scores)[:top_k],
				list(sorted_ids)[:top_k],
			)

		reranked_contents, reranked_scores, reranked_ids = zip(
			*map(
				sort_row,
				contents_list,
				scores_list,
				ids_list,
				time_list,
				[top_k] * len(contents_list),
			)
		)

		return list(reranked_contents), list(reranked_ids), list(reranked_scores)
