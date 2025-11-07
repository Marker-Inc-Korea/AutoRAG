from typing import List, Tuple, Iterable

import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils.util import (
	make_batch,
	sort_by_scores,
	flatten_apply,
	select_top_k,
	pop_params,
	result_to_dataframe,
	empty_cuda_cache,
)


class FlagEmbeddingReranker(BasePassageReranker):
	def __init__(
		self, project_dir, model_name: str = "BAAI/bge-reranker-large", *args, **kwargs
	):
		"""
		Initialize the FlagEmbeddingReranker module.

		:param project_dir: The project directory.
		:param model_name: The name of the BAAI Reranker normal-model name.
		Default is "BAAI/bge-reranker-large"
		:param kwargs: Extra parameter for FlagEmbedding.FlagReranker
		"""
		super().__init__(project_dir)
		try:
			from FlagEmbedding import FlagReranker
		except ImportError:
			raise ImportError(
				"FlagEmbeddingReranker requires the 'FlagEmbedding' package to be installed."
			)
		model_params = pop_params(FlagReranker.__init__, kwargs)
		model_params.pop("model_name_or_path", None)
		self.model = FlagReranker(model_name_or_path=model_name, **model_params)

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
		Rerank a list of contents based on their relevance to a query using BAAI normal-Reranker model.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param batch: The number of queries to be processed in a batch
			Default is 64.
		:return: Tuple of lists containing the reranked contents, ids, and scores
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


def flag_embedding_run_model(input_texts, model, batch_size: int):
	try:
		import torch
	except ImportError:
		raise ImportError("FlagEmbeddingReranker requires PyTorch to be installed.")
	batch_input_texts = make_batch(input_texts, batch_size)
	results = []
	for batch_texts in batch_input_texts:
		with torch.no_grad():
			pred_scores = model.compute_score(sentence_pairs=batch_texts)
		if not isinstance(pred_scores, Iterable):
			results.append(pred_scores)
		else:
			results.extend(pred_scores)
	return results
