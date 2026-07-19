from typing import List, Tuple

import numpy as np
import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils.util import (
	make_batch,
	sort_by_scores,
	flatten_apply,
	select_top_k,
	result_to_dataframe,
	empty_cuda_cache,
)


class KoReranker(BasePassageReranker):
	def __init__(self, project_dir: str, *args, **kwargs):
		super().__init__(project_dir)
		try:
			import torch
			from transformers import AutoModelForSequenceClassification, AutoTokenizer
		except ImportError:
			raise ImportError("For using KoReranker, please install torch first.")

		model_path = "Dongjin-kr/ko-reranker"
		self.tokenizer = AutoTokenizer.from_pretrained(model_path)
		self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
		self.model.eval()
		# Determine the device to run the model on (GPU if available, otherwise CPU)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)

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
		Rerank a list of contents based on their relevance to a query using ko-reranker.
		ko-reranker is a reranker based on korean (https://huggingface.co/Dongjin-kr/ko-reranker).

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
		scores_nps = flatten_apply(
			koreranker_run_model,
			nested_list,
			model=self.model,
			batch_size=batch,
			tokenizer=self.tokenizer,
			device=self.device,
		)

		rerank_scores = list(
			map(
				lambda scores: exp_normalize(np.array(scores)).astype(float), scores_nps
			)
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


def koreranker_run_model(input_texts, model, tokenizer, device, batch_size: int):
	try:
		import torch
	except ImportError:
		raise ImportError("For using KoReranker, please install torch first.")
	batch_input_texts = make_batch(input_texts, batch_size)
	results = []
	for batch_texts in batch_input_texts:
		inputs = tokenizer(
			batch_texts,
			padding=True,
			truncation=True,
			return_tensors="pt",
			max_length=512,
		)
		inputs = inputs.to(device)
		with torch.no_grad():
			scores = (
				model(**inputs, return_dict=True)
				.logits.view(
					-1,
				)
				.float()
			)
			scores_np = scores.cpu().numpy()
			results.extend(scores_np)
	return results


def exp_normalize(x):
	b = x.max()
	y = np.exp(x - b)
	return y / y.sum()
