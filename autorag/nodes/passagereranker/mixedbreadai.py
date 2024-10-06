import os
from typing import List, Tuple

import pandas as pd
from mixedbread_ai.client import AsyncMixedbreadAI

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils.util import (
	result_to_dataframe,
	get_event_loop,
	process_batch,
	pop_params,
)


class MixedbreadAIReranker(BasePassageReranker):
	def __init__(
		self,
		project_dir: str,
		*args,
		**kwargs,
	):
		"""
		Initialize mixedbread-ai rerank node.

		:param project_dir: The project directory path.
		:param api_key: The API key for MixedbreadAI rerank.
		    You can set it in the environment variable MXBAI_API_KEY.
		    Or, you can directly set it on the config YAML file using this parameter.
		    Default is env variable "MXBAI_API_KEY".
		:param kwargs: Extra arguments that are not affected
		"""
		super().__init__(project_dir)
		api_key = kwargs.pop("api_key", None)
		api_key = os.getenv("MXBAI_API_KEY", None) if api_key is None else api_key
		if api_key is None:
			raise KeyError(
				"Please set the API key for Mixedbread AI rerank in the environment variable MXBAI_API_KEY "
				"or directly set it on the config YAML file."
			)
		self.client = AsyncMixedbreadAI(api_key=api_key)

	def __del__(self):
		del self.client
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, scores, ids = self.cast_to_run(previous_result)
		top_k = kwargs.pop("top_k")
		batch = kwargs.pop("batch", 8)
		model = kwargs.pop("model", "mixedbread-ai/mxbai-rerank-large-v1")
		rerank_params = pop_params(self.client.reranking, kwargs)
		return self._pure(queries, contents, ids, top_k, model, batch, **rerank_params)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		ids_list: List[List[str]],
		top_k: int,
		model: str = "mixedbread-ai/mxbai-rerank-large-v1",
		batch: int = 8,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents with mixedbread-ai rerank models.
		You can get the API key from https://www.mixedbread.ai/api-reference#quick-start-guide and set it in the environment variable MXBAI_API_KEY.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param model: The model name for mixedbread-ai rerank.
			You can choose between "mixedbread-ai/mxbai-rerank-large-v1", "mixedbread-ai/mxbai-rerank-base-v1" and "mixedbread-ai/mxbai-rerank-xsmall-v1".
			Default is "mixedbread-ai/mxbai-rerank-large-v1".
		:param batch: The number of queries to be processed in a batch
		        :return: Tuple of lists containing the reranked contents, ids, and scores
		"""
		tasks = [
			mixedbreadai_rerank_pure(
				self.client, query, contents, ids, top_k=top_k, model=model
			)
			for query, contents, ids in zip(queries, contents_list, ids_list)
		]
		loop = get_event_loop()
		results = loop.run_until_complete(process_batch(tasks, batch))

		content_result, id_result, score_result = zip(*results)

		return list(content_result), list(id_result), list(score_result)


async def mixedbreadai_rerank_pure(
	client: AsyncMixedbreadAI,
	query: str,
	documents: List[str],
	ids: List[str],
	top_k: int,
	model: str = "mixedbread-ai/mxbai-rerank-large-v1",
) -> Tuple[List[str], List[str], List[float]]:
	"""
	Rerank a list of contents with mixedbread-ai rerank models.

	:param client: The mixedbread-ai client to use for reranking
	:param query: The query to use for reranking
	:param documents: The list of contents to rerank
	:param ids: The list of ids corresponding to the documents
	:param top_k: The number of passages to be retrieved
	:param model: The model name for mixedbread-ai rerank.
	    You can choose between "mixedbread-ai/mxbai-rerank-large-v1" and "mixedbread-ai/mxbai-rerank-base-v1".
	    Default is "mixedbread-ai/mxbai-rerank-large-v1".
	:return: Tuple of lists containing the reranked contents, ids, and scores
	"""

	results = await client.reranking(
		query=query,
		input=documents,
		top_k=top_k,
		model=model,
	)
	reranked_scores: List[float] = list(map(lambda x: x.score, results.data))
	reranked_scores_float = list(map(float, reranked_scores))
	indices = list(map(lambda x: x.index, results.data))
	reranked_contents = list(map(lambda x: documents[x], indices))
	reranked_ids: List[str] = list(map(lambda i: ids[i], indices))
	return reranked_contents, reranked_ids, reranked_scores_float
