from typing import List

import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils import result_to_dataframe


class PassReranker(BasePassageReranker):
	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		top_k = kwargs.pop("top_k")

		_, contents_list, scores_list, ids_list = self.cast_to_run(previous_result)
		return self._pure(contents_list, scores_list, ids_list, top_k)

	def _pure(
		self,
		contents_list: List[List[str]],
		scores_list: List[List[float]],
		ids_list: List[List[str]],
		top_k: int,
	):
		"""
		Do not perform reranking.
		Return the given top-k passages as is.
		"""
		contents_list = list(map(lambda x: x[:top_k], contents_list))
		scores_list = list(map(lambda x: x[:top_k], scores_list))
		ids_list = list(map(lambda x: x[:top_k], ids_list))
		return contents_list, ids_list, scores_list
