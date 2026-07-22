from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from autorag.embedding.base import EmbeddingModel
from autorag.evaluation.metric.util import calculate_cosine_similarity
from autorag.nodes.passagefilter.base import BasePassageFilter
from autorag.nodes.passagefilter.similarity_threshold_cutoff import (
	embedding_query_content,
)
from autorag.utils import result_to_dataframe
from autorag.utils.util import empty_cuda_cache, pop_params


class SimilarityPercentileCutoff(BasePassageFilter):
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs):
		"""
		Initialize the SimilarityPercentileCutoff module

		:param project_dir: The project directory to use for initializing the module
		:param embedding_model: The embedding model string to use for calculating similarity
		        Default is "openai" which is OpenAI text-embedding-ada-002 embedding model.
		"""
		super().__init__(project_dir, *args, **kwargs)
		embedding_model = kwargs.pop("embedding_model", "openai")
		self.embedding_model = EmbeddingModel.load(embedding_model)()

	def __del__(self):
		super().__del__()
		del self.embedding_model

		empty_cuda_cache()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, **kwargs):
		queries, contents, scores, ids = self.cast_to_run(previous_result)
		kwargs = pop_params(self._pure, kwargs)
		return self._pure(queries, contents, scores, ids, **kwargs)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		scores_list: List[List[float]],
		ids_list: List[List[str]],
		percentile: float,
		batch: int = 128,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Re-calculate each content's similarity with the query and filter out the contents that are below the content's
		length times percentile. If This is a filter and does not override scores. The output of scores is not coming from
		query-content similarity.
		If the value of content's length times percentile is less than 1, keep the only one highest similarity content.

		:param queries: The list of queries to use for filtering
		:param contents_list: The list of lists of contents to filter
		:param scores_list: The list of lists of scores retrieved
		:param ids_list: The list of lists of ids retrieved
		:param percentile: The percentile to cut off
		:param batch: The number of queries to be processed in a batch
		    Default is 128.
		:return: Tuple of lists containing the filtered contents, ids, and scores
		"""
		query_embeddings, content_embeddings = embedding_query_content(
			queries, contents_list, self.embedding_model, batch
		)

		results = list(
			map(
				lambda x: self.__row_pure(x[0], x[1], x[2], x[3], x[4], percentile),
				zip(
					query_embeddings,
					content_embeddings,
					contents_list,
					ids_list,
					scores_list,
				),
			)
		)

		remain_content_list = list(map(lambda x: x[0], results))
		remain_ids_list = list(map(lambda x: x[1], results))
		remain_scores_list = list(map(lambda x: x[2], results))

		return remain_content_list, remain_ids_list, remain_scores_list

	@staticmethod
	def __row_pure(
		query_embedding: str,
		content_embeddings: List[List[float]],
		content_list: List[str],
		ids_list: List[str],
		scores_list: List[float],
		percentile: float,
	) -> Tuple[List[str], List[str], List[float]]:
		"""
		Return tuple of lists containing the filtered contents, ids, and scores

		:param query_embedding: Query embedding
		:param content_embeddings: Each content embedding
		:param content_list: Each content
		:param ids_list: Each id
		:param scores_list: Each score
		:param percentile: The percentile to cut off
		:return: Tuple of lists containing the filtered contents, ids, and scores
		"""
		num_top_k = int(len(content_embeddings) * percentile)

		if num_top_k == 0:
			num_top_k = 1

		similarities = np.array(
			list(
				map(
					lambda x: calculate_cosine_similarity(query_embedding, x),
					content_embeddings,
				)
			)
		).tolist()

		content_id_score_similarity = list(
			zip(ids_list, content_list, scores_list, similarities)
		)

		sorted_content_id_score_similarity = sorted(
			content_id_score_similarity, key=lambda x: x[3], reverse=True
		)[:num_top_k]

		content_result, id_result, score_result, _ = zip(
			*sorted_content_id_score_similarity
		)
		return list(content_result), list(id_result), list(score_result)
