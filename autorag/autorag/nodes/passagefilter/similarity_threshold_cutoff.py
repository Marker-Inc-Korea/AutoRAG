from typing import List, Tuple

import numpy as np
import pandas as pd

from autorag.embedding.base import EmbeddingModel
from autorag.evaluation.metric.util import calculate_cosine_similarity
from autorag.nodes.passagefilter.base import BasePassageFilter
from autorag.utils.util import (
	embedding_query_content,
	empty_cuda_cache,
	result_to_dataframe,
	pop_params,
)


class SimilarityThresholdCutoff(BasePassageFilter):
	def __init__(self, project_dir: str, *args, **kwargs):
		"""
		Initialize the SimilarityThresholdCutoff module

		:param project_dir: The project directory to use for initializing the module
		:param embedding_model: The embedding model string to use for calculating similarity
		        Default is "openai" which is OpenAI text-embedding-ada-002 embedding model.
		"""
		super().__init__(project_dir, *args, **kwargs)
		embedding_model = kwargs.get("embedding_model", "openai")
		self.embedding_model = EmbeddingModel.load(embedding_model)()

	def __del__(self):
		del self.embedding_model
		empty_cuda_cache()
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		kwargs = pop_params(self._pure, kwargs)
		queries, contents, scores, ids = self.cast_to_run(previous_result)
		return self._pure(queries, contents, scores, ids, *args, **kwargs)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		scores_list: List[List[float]],
		ids_list: List[List[str]],
		threshold: float,
		batch: int = 128,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Re-calculate each content's similarity with the query and filter out the contents that are below the threshold.
		If all contents are filtered, keep the only one highest similarity content.
		This is a filter and does not override scores.
		The output of scores is not coming from query-content similarity.

		:param queries: The list of queries to use for filtering
		:param contents_list: The list of lists of contents to filter
		:param scores_list: The list of lists of scores retrieved
		:param ids_list: The list of lists of ids retrieved
		:param threshold: The threshold to cut off
		:param batch: The number of queries to be processed in a batch
		    Default is 128.
		:return: Tuple of lists containing the filtered contents, ids, and scores
		"""
		query_embeddings, content_embeddings = embedding_query_content(
			queries, contents_list, self.embedding_model, batch
		)

		remain_indices = list(
			map(
				lambda x: self.__row_pure(x[0], x[1], threshold),
				zip(query_embeddings, content_embeddings),
			)
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

	@staticmethod
	def __row_pure(
		query_embedding: str, content_embeddings: List[List[float]], threshold: float
	) -> List[int]:
		"""
		Return indices that have to remain.
		Return at least one index if there is nothing to remain.

		:param query_embedding: Query embedding
		:param content_embeddings: Each content embedding
		:param threshold: The threshold to cut off
		:return: Indices to remain at the contents
		"""

		similarities = np.array(
			list(
				map(
					lambda x: calculate_cosine_similarity(query_embedding, x),
					content_embeddings,
				)
			)
		)
		result = np.where(similarities >= threshold)[0].tolist()
		if len(result) > 0:
			return result
		return [np.argmax(similarities)]
