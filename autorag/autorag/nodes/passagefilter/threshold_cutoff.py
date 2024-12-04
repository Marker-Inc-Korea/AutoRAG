from typing import List, Tuple

import pandas as pd

from autorag.nodes.passagefilter.base import BasePassageFilter
from autorag.utils.util import convert_inputs_to_list, result_to_dataframe


class ThresholdCutoff(BasePassageFilter):
	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		_, contents, scores, ids = self.cast_to_run(previous_result)
		return self._pure(contents, scores, ids, *args, **kwargs)

	def _pure(
		self,
		contents_list: List[List[str]],
		scores_list: List[List[float]],
		ids_list: List[List[str]],
		threshold: float,
		reverse: bool = False,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Filters the contents, scores, and ids based on a previous result's score.
		Keeps at least one item per query if all scores are below the threshold.

		:param contents_list: List of content strings for each query.
		:param scores_list: List of scores for each content.
		:param ids_list: List of ids for each content.
		:param threshold: The minimum score to keep an item.
		:param reverse: If True, the lower the score, the better.
		    Default is False.
		:return: Filtered lists of contents, ids, and scores.
		"""
		remain_indices = list(
			map(lambda x: self.__row_pure(x, threshold, reverse), scores_list)
		)

		remain_content_list = list(
			map(lambda c, idx: [c[i] for i in idx], contents_list, remain_indices)
		)
		remain_scores_list = list(
			map(lambda s, idx: [s[i] for i in idx], scores_list, remain_indices)
		)
		remain_ids_list = list(
			map(lambda _id, idx: [_id[i] for i in idx], ids_list, remain_indices)
		)

		return remain_content_list, remain_ids_list, remain_scores_list

	@convert_inputs_to_list
	def __row_pure(
		self, scores_list: List[float], threshold: float, reverse: bool = False
	) -> List[int]:
		"""
		Return indices that have to remain.
		Return at least one index if there is nothing to remain.

		:param scores_list: Each score
		:param threshold: The threshold to cut off
		:param reverse: If True, the lower the score, the better
			Default is False.
		:return: Indices to remain at the contents
		"""
		assert isinstance(scores_list, list), "scores_list must be a list."

		if reverse:
			remain_indices = [
				i for i, score in enumerate(scores_list) if score <= threshold
			]
			default_index = scores_list.index(min(scores_list))
		else:
			remain_indices = [
				i for i, score in enumerate(scores_list) if score >= threshold
			]
			default_index = scores_list.index(max(scores_list))

		return remain_indices if remain_indices else [default_index]
