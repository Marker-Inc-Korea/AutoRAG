import os
from typing import List, Tuple
import pandas as pd
import voyageai

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils.util import result_to_dataframe


class VoyageAIReranker(BasePassageReranker):
	def __init__(self, project_dir: str, *args, **kwargs):
		super().__init__(project_dir)
		api_key = kwargs.pop("api_key", None)
		api_key = os.getenv("VOYAGE_API_KEY", None) if api_key is None else api_key
		if api_key is None:
			raise KeyError(
				"Please set the API key for VoyageAI rerank in the environment variable VOYAGE_API_KEY "
				"or directly set it on the config YAML file."
			)

		self.voyage_client = voyageai.Client()

	def __del__(self):
		del self.voyage_client
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, scores, ids = self.cast_to_run(previous_result)
		top_k = kwargs.pop("top_k")
		batch = kwargs.pop("batch", 64)
		model = kwargs.pop("model", "rerank-2")
		truncation = kwargs.pop("truncation", True)
		return self._pure(queries, contents, ids, top_k, batch, model, truncation)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		ids_list: List[List[str]],
		top_k: int,
		batch: int = 64,
		model: str = "rerank-2",
		truncation: bool = True,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents with VoyageAI rerank models.
		You can get the API key from https://docs.voyageai.com/docs/api-key-and-installation and set it in the environment variable VOYAGE_API_KEY.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param batch: The number of queries to be processed in a batch
		:param model: The model name for Cohere rerank.
		    You can choose between "rerank-2" and "rerank-2-lite".
		    Default is "rerank-2".
		:param truncation: Whether to truncate the input to satisfy the 'context length limit' on the query and the documents.
		:return: Tuple of lists containing the reranked contents, ids, and scores
		"""
		content_result, id_result, score_result = zip(
			*[
				voyageai_rerank_pure(
					self.voyage_client, model, query, document, ids, top_k, truncation
				)
				for query, document, ids in zip(queries, contents_list, ids_list)
			]
		)

		return content_result, id_result, score_result


def voyageai_rerank_pure(
	voyage_client: voyageai.Client,
	model: str,
	query: str,
	documents: List[str],
	ids: List[str],
	top_k: int,
	truncation: bool = True,
) -> Tuple[List[str], List[str], List[float]]:
	"""
	Rerank a list of contents with Cohere rerank models.

	:param voyage_client: The Voyage Client to use for reranking
	:param model: The model name for Cohere rerank
	:param query: The query to use for reranking
	:param documents: The list of contents to rerank
	:param ids: The list of ids corresponding to the documents
	:param top_k: The number of passages to be retrieved
	:param truncation: Whether to truncate the input to satisfy the 'context length limit' on the query and the documents.
	:return: Tuple of lists containing the reranked contents, ids, and scores
	"""
	rerank_results = voyage_client.rerank(
		model=model,
		query=query,
		documents=documents,
		top_k=top_k,
		truncation=truncation,
	)
	reranked_scores: List[float] = list(
		map(lambda x: x.relevance_score, rerank_results.results)
	)
	reranked_contents: List[str] = list(
		map(lambda x: x.document, rerank_results.results)
	)
	indices = list(map(lambda x: x.index, rerank_results.results))
	reranked_ids: List[str] = list(map(lambda i: ids[i], indices))
	return reranked_contents, reranked_ids, reranked_scores
