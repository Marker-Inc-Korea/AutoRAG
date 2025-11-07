import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

from autorag.nodes.passagefilter.base import BasePassageFilter
from autorag.utils import fetch_contents, result_to_dataframe

logger = logging.getLogger("AutoRAG")


class RecencyFilter(BasePassageFilter):
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs):
		super().__init__(project_dir, *args, **kwargs)
		self.corpus_df = pd.read_parquet(
			os.path.join(project_dir, "data", "corpus.parquet"), engine="pyarrow"
		)

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		_, contents, scores, ids = self.cast_to_run(previous_result, *args, **kwargs)
		metadatas = fetch_contents(self.corpus_df, ids, column_name="metadata")
		times = [
			[time["last_modified_datetime"] for time in time_list]
			for time_list in metadatas
		]
		return self._pure(contents, scores, ids, times, *args, **kwargs)

	def _pure(
		self,
		contents_list: List[List[str]],
		scores_list: List[List[float]],
		ids_list: List[List[str]],
		time_list: List[List[datetime]],
		threshold_datetime: Union[datetime, date],
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Filter out the contents that are below the threshold datetime.
		If all contents are filtered, keep the only one recency content.
		If the threshold date format is incorrect, return the original contents.

		:param contents_list: The list of lists of contents to filter
		:param scores_list: The list of lists of scores retrieved
		:param ids_list: The list of lists of ids retrieved
		:param time_list: The list of lists of datetime retrieved
		:param threshold_datetime: The threshold to cut off.
			In recency filter, you have to use the datetime.datetime object or datetime.date object.
			All you need to do is to set the date at your YAML file.
			For example, you can write "2010-09-09 3:45:06" or "2010-09-09" in the YAML file.
		:return: Tuple of lists containing the filtered contents, ids, and scores
		"""
		if not (
			isinstance(threshold_datetime, datetime)
			or isinstance(threshold_datetime, date)
		):
			raise ValueError(
				f"Threshold should be a datetime object, but got {type(threshold_datetime)}"
			)

		if not isinstance(threshold_datetime, datetime):
			threshold_datetime = datetime.combine(
				threshold_datetime, datetime.min.time()
			)

		time_list = [
			list(
				map(
					lambda t: datetime.combine(t, datetime.min.time())
					if not isinstance(t, datetime)
					else t,
					time,
				)
			)
			for time in time_list
		]

		def sort_row(contents, scores, ids, time, _datetime_threshold):
			combined = list(zip(contents, scores, ids, time))
			combined_filtered = [
				item for item in combined if item[3] >= _datetime_threshold
			]

			if combined_filtered:
				remain_contents, remain_scores, remain_ids, _ = zip(*combined_filtered)
			else:
				combined.sort(key=lambda x: x[3], reverse=True)
				remain_contents, remain_scores, remain_ids, _ = zip(*combined[:1])

			return list(remain_contents), list(remain_ids), list(remain_scores)

		remain_contents_list, remain_ids_list, remain_scores_list = zip(
			*map(
				sort_row,
				contents_list,
				scores_list,
				ids_list,
				time_list,
				[threshold_datetime] * len(contents_list),
			)
		)

		return remain_contents_list, remain_ids_list, remain_scores_list
