from typing import List, Tuple

import numpy as np
import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils.util import (
	flatten_apply,
	sort_by_scores,
	select_top_k,
	pop_params,
	result_to_dataframe,
	empty_cuda_cache,
)


class ColbertReranker(BasePassageReranker):
	def __init__(
		self,
		project_dir: str,
		model_name: str = "colbert-ir/colbertv2.0",
		*args,
		**kwargs,
	):
		"""
		Initialize a colbert rerank model for reranking.

		:param project_dir: The project directory
		:param model_name: The model name for Colbert rerank.
			You can choose a colbert model for reranking.
			The default is "colbert-ir/colbertv2.0".
		:param kwargs: Extra parameter for the model.
		"""
		super().__init__(project_dir)
		try:
			import torch
			from transformers import AutoModel, AutoTokenizer
		except ImportError:
			raise ImportError(
				"Pytorch is not installed. Please install pytorch to use Colbert reranker."
			)
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		model_params = pop_params(AutoModel.from_pretrained, kwargs)
		self.model = AutoModel.from_pretrained(model_name, **model_params).to(
			self.device
		)
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
		Rerank a list of contents with Colbert rerank models.
		You can get more information about a Colbert model at https://huggingface.co/colbert-ir/colbertv2.0.
		It uses BERT-based model, so recommend using CUDA gpu for faster reranking.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param batch: The number of queries to be processed in a batch
			Default is 64.

		:return: Tuple of lists containing the reranked contents, ids, and scores
		"""

		# get query and content embeddings
		query_embedding_list = get_colbert_embedding_batch(
			queries, self.model, self.tokenizer, batch
		)
		content_embedding_list = flatten_apply(
			get_colbert_embedding_batch,
			contents_list,
			model=self.model,
			tokenizer=self.tokenizer,
			batch_size=batch,
		)
		df = pd.DataFrame(
			{
				"ids": ids_list,
				"query_embedding": query_embedding_list,
				"contents": contents_list,
				"content_embedding": content_embedding_list,
			}
		)
		temp_df = df.explode("content_embedding")
		temp_df["score"] = temp_df.apply(
			lambda x: get_colbert_score(x["query_embedding"], x["content_embedding"]),
			axis=1,
		)
		df["scores"] = (
			temp_df.groupby(level=0, sort=False)["score"].apply(list).tolist()
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


def get_colbert_embedding_batch(
	input_strings: List[str], model, tokenizer, batch_size: int
) -> List[np.array]:
	try:
		import torch
	except ImportError:
		raise ImportError(
			"Pytorch is not installed. Please install pytorch to use Colbert reranker."
		)
	encoding = tokenizer(
		input_strings,
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=model.config.max_position_embeddings,
	)

	input_batches = slice_tokenizer_result(encoding, batch_size)
	result_embedding = []
	with torch.no_grad():
		for encoding_batch in input_batches:
			result_embedding.append(model(**encoding_batch).last_hidden_state)
	total_tensor = torch.cat(
		result_embedding, dim=0
	)  # shape [batch_size, token_length, embedding_dim]
	tensor_results = list(total_tensor.chunk(total_tensor.size()[0]))

	if torch.cuda.is_available():
		return list(map(lambda x: x.detach().cpu().numpy(), tensor_results))
	else:
		return list(map(lambda x: x.detach().numpy(), tensor_results))


def slice_tokenizer_result(tokenizer_output, batch_size):
	input_ids_batches = slice_tensor(tokenizer_output["input_ids"], batch_size)
	attention_mask_batches = slice_tensor(
		tokenizer_output["attention_mask"], batch_size
	)
	token_type_ids_batches = slice_tensor(
		tokenizer_output.get("token_type_ids", None), batch_size
	)
	return [
		{
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"token_type_ids": token_type_ids,
		}
		for input_ids, attention_mask, token_type_ids in zip(
			input_ids_batches, attention_mask_batches, token_type_ids_batches
		)
	]


def slice_tensor(input_tensor, batch_size):
	try:
		import torch
	except ImportError:
		raise ImportError(
			"Pytorch is not installed. Please install pytorch to use Colbert reranker."
		)
	# Calculate the number of full batches
	num_full_batches = input_tensor.size(0) // batch_size

	# Slice the tensor into batches
	tensor_list = [
		input_tensor[i * batch_size : (i + 1) * batch_size]
		for i in range(num_full_batches)
	]

	# Handle the last batch if it's smaller than batch_size
	remainder = input_tensor.size(0) % batch_size
	if remainder:
		tensor_list.append(input_tensor[-remainder:])

	device = "cuda" if torch.cuda.is_available() else "cpu"
	tensor_list = list(map(lambda x: x.to(device), tensor_list))

	return tensor_list


def get_colbert_score(query_embedding: np.array, content_embedding: np.array) -> float:
	if query_embedding.ndim == 3 and content_embedding.ndim == 3:
		query_embedding = query_embedding.reshape(-1, query_embedding.shape[-1])
		content_embedding = content_embedding.reshape(-1, content_embedding.shape[-1])

	sim_matrix = np.dot(query_embedding, content_embedding.T) / (
		np.linalg.norm(query_embedding, axis=1)[:, np.newaxis]
		* np.linalg.norm(content_embedding, axis=1)
	)
	max_sim_scores = np.max(sim_matrix, axis=1)
	return float(np.mean(max_sim_scores))
