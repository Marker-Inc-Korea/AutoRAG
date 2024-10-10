from typing import List, Tuple

import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.nodes.passagereranker.flag_embedding import flag_embedding_run_model
from autorag.utils.util import (
	flatten_apply,
	sort_by_scores,
	select_top_k,
	pop_params,
	result_to_dataframe,
	empty_cuda_cache,
)


class FlagEmbeddingLLMReranker(BasePassageReranker):
	def __init__(
		self,
		project_dir,
		model_name: str = "BAAI/bge-reranker-v2-gemma",
		*args,
		**kwargs,
	):
		"""
		Initialize the FlagEmbeddingReranker module.

		:param project_dir: The project directory.
		:param model_name: The name of the BAAI Reranker LLM-based-model name.
		Default is "BAAI/bge-reranker-v2-gemma"
		:param kwargs: Extra parameter for FlagEmbedding.FlagReranker
		"""
		super().__init__(project_dir)
		try:
			from FlagEmbedding import FlagLLMReranker
		except ImportError:
			raise ImportError(
				"FlagEmbeddingLLMReranker requires the 'FlagEmbedding' package to be installed."
			)
		model_params = pop_params(FlagLLMReranker.__init__, kwargs)
		model_params.pop("model_name_or_path", None)
		self.model = FlagLLMReranker(model_name_or_path=model_name, **model_params)

	def __del__(self):
		del self.model
		empty_cuda_cache()
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, _, ids = self.cast_to_run(previous_result)
		top_k = kwargs.pop("top_k")
		batch = kwargs.pop("batch", 64)
		return self._pure(queries, contents, ids, top_k, batch)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		ids_list: List[List[str]],
		top_k: int,
		batch: int = 64,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents based on their relevance to a query using BAAI LLM-based-Reranker model.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param batch: The number of queries to be processed in a batch
			Default is 64.

		:return: tuple of lists containing the reranked contents, ids, and scores
		"""

		nested_list = [
			list(map(lambda x: [query, x], content_list))
			for query, content_list in zip(queries, contents_list)
		]
		rerank_scores = flatten_apply(
			flag_embedding_run_model, nested_list, model=self.model, batch_size=batch
		)

		df = pd.DataFrame(
			{
				"contents": contents_list,
				"ids": ids_list,
				"scores": rerank_scores,
			}
		)
		df[["contents", "ids", "scores"]] = df.apply(
			sort_by_scores, axis=1, result_type="expand"
		)
		results = select_top_k(df, ["contents", "ids", "scores"], top_k)

		return (
			results["contents"].tolist(),
			results["ids"].tolist(),
			results["scores"].tolist(),
		)
