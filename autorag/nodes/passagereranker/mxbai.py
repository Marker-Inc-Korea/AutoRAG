from typing import List, Tuple

import pandas as pd
from sentence_transformers import CrossEncoder

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils.util import result_to_dataframe


class MxBaiReranker(BasePassageReranker):
	def __init__(
		self,
		project_dir: str,
		model_name: str = "mixedbread-ai/mxbai-rerank-large-v1",
		*args,
		**kwargs,
	):
		"""
		Initialize mixedbread-ai rerank node.

		:param project_dir: The project directory path.
		:param model_name: The name of the mixedbread-ai model to use for reranking
			Note: default model name is √
		:param kwargs: Extra arguments that are not affected
		"""
		super().__init__(project_dir)
		self.model = CrossEncoder(model_name)

	def __del__(self):
		del self.model
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, scores, ids = self.cast_to_run(previous_result)
		top_k = kwargs.pop("top_k")
		return self._pure(queries, contents, scores, ids, top_k)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		scores_list: List[List[float]],
		ids_list: List[List[str]],
		top_k: int,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents with mixedbread-ai rerank models.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param scores_list: The list of lists of scores retrieved from the initial ranking
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:return: Tuple of lists containing the reranked contents, ids, and scores
		"""
		content_result, id_result, score_result = zip(
			*[
				mxbai_rerank_pure(self.model, query, document, ids, top_k)
				for query, document, ids in zip(queries, contents_list, ids_list)
			]
		)

		return content_result, id_result, score_result


def mxbai_rerank_pure(
	model: CrossEncoder,
	query: str,
	documents: List[str],
	ids: List[str],
	top_k: int,
) -> Tuple[List[str], List[str], List[float]]:
	"""
	Rerank a list of contents with mixedbread-ai rerank models.

	:param model: The model to use for reranking.
	It should be a CrossEncoder model
	:param query: The query to use for reranking
	:param documents: The list of contents to rerank
	:param ids: The list of ids corresponding to the documents
	:param top_k: The number of passages to be retrieved
	:return: Tuple of lists containing the reranked contents, ids, and scores
	"""

	results = model.rank(query, documents, return_documents=True, top_k=top_k)
	reranked_scores: List[float] = list(map(lambda x: x["score"], results))
	reranked_scores_float = list(map(float, reranked_scores))
	indices = list(map(lambda x: x["corpus_id"], results))
	reranked_contents: List[str] = list(map(lambda i: documents[i], indices))
	reranked_ids: List[str] = list(map(lambda i: ids[i], indices))
	return reranked_contents, reranked_ids, reranked_scores_float
