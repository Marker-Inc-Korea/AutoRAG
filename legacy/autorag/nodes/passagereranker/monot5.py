from itertools import chain
from typing import List, Tuple

import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils.util import (
	make_batch,
	sort_by_scores,
	flatten_apply,
	select_top_k,
	result_to_dataframe,
	pop_params,
	empty_cuda_cache,
)

prediction_tokens = {
	"castorini/monot5-base-msmarco": ["▁false", "▁true"],
	"castorini/monot5-base-msmarco-10k": ["▁false", "▁true"],
	"castorini/monot5-large-msmarco": ["▁false", "▁true"],
	"castorini/monot5-large-msmarco-10k": ["▁false", "▁true"],
	"castorini/monot5-base-med-msmarco": ["▁false", "▁true"],
	"castorini/monot5-3b-med-msmarco": ["▁false", "▁true"],
	"castorini/monot5-3b-msmarco-10k": ["▁false", "▁true"],
	"unicamp-dl/mt5-base-en-msmarco": ["▁no", "▁yes"],
	"unicamp-dl/ptt5-base-pt-msmarco-10k-v2": ["▁não", "▁sim"],
	"unicamp-dl/ptt5-base-pt-msmarco-100k-v2": ["▁não", "▁sim"],
	"unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2": ["▁não", "▁sim"],
	"unicamp-dl/mt5-base-en-pt-msmarco-v2": ["▁no", "▁yes"],
	"unicamp-dl/mt5-base-mmarco-v2": ["▁no", "▁yes"],
	"unicamp-dl/mt5-base-en-pt-msmarco-v1": ["▁no", "▁yes"],
	"unicamp-dl/mt5-base-mmarco-v1": ["▁no", "▁yes"],
	"unicamp-dl/ptt5-base-pt-msmarco-10k-v1": ["▁não", "▁sim"],
	"unicamp-dl/ptt5-base-pt-msmarco-100k-v1": ["▁não", "▁sim"],
	"unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1": ["▁não", "▁sim"],
	"unicamp-dl/mt5-3B-mmarco-en-pt": ["▁", "▁true"],
	"unicamp-dl/mt5-13b-mmarco-100k": ["▁", "▁true"],
}


class MonoT5(BasePassageReranker):
	def __init__(
		self,
		project_dir: str,
		model_name: str = "castorini/monot5-3b-msmarco-10k",
		*args,
		**kwargs,
	):
		"""
		Initialize the MonoT5 reranker.

		:param project_dir: The project directory
		:param model_name: The name of the MonoT5 model to use for reranking
			Note: default model name is 'castorini/monot5-3b-msmarco-10k'
				If there is a '/' in the model name parameter,
				when we create the file to store the results, the path will be twisted because of the '/'.
				Therefore, it will be received as '_' instead of '/'.
		:param kwargs: The extra arguments for the MonoT5 reranker
		"""
		super().__init__(project_dir)
		try:
			import torch
			from transformers import T5Tokenizer, T5ForConditionalGeneration
		except ImportError:
			raise ImportError("For using MonoT5 Reranker, please install torch first.")
		# replace '_' to '/'
		if "_" in model_name:
			model_name = model_name.replace("_", "/")
		# Load the tokenizer and model from the pre-trained MonoT5 model
		self.tokenizer = T5Tokenizer.from_pretrained(model_name)
		model_params = pop_params(T5ForConditionalGeneration.from_pretrained, kwargs)
		self.model = T5ForConditionalGeneration.from_pretrained(
			model_name, **model_params
		).eval()

		# Determine the device to run the model on (GPU if available, otherwise CPU)
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model.to(self.device)

		token_false, token_true = prediction_tokens[model_name]
		self.token_false_id = self.tokenizer.convert_tokens_to_ids(token_false)
		self.token_true_id = self.tokenizer.convert_tokens_to_ids(token_true)

	def __del__(self):
		del self.model
		del self.tokenizer
		empty_cuda_cache()
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, _, ids = self.cast_to_run(previous_result)
		top_k = kwargs.get("top_k", 3)
		batch = kwargs.get("batch", 64)
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
		Rerank a list of contents based on their relevance to a query using MonoT5.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved

		:param batch: The number of queries to be processed in a batch
		:return: tuple of lists containing the reranked contents, ids, and scores
		"""
		# Retrieve the tokens used by the model to represent false and true predictions

		nested_list = [
			list(map(lambda x: [f"Query: {query} Document: {x}"], content_list))
			for query, content_list in zip(queries, contents_list)
		]

		rerank_scores = flatten_apply(
			monot5_run_model,
			nested_list,
			model=self.model,
			batch_size=batch,
			tokenizer=self.tokenizer,
			device=self.device,
			token_false_id=self.token_false_id,
			token_true_id=self.token_true_id,
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


def monot5_run_model(
	input_texts,
	model,
	batch_size: int,
	tokenizer,
	device,
	token_false_id,
	token_true_id,
):
	try:
		import torch
	except ImportError:
		raise ImportError("For using MonoT5 Reranker, please install torch first.")
	batch_input_texts = make_batch(input_texts, batch_size)
	results = []
	for batch_texts in batch_input_texts:
		flattened_batch_texts = list(chain.from_iterable(batch_texts))
		input_encodings = tokenizer(
			flattened_batch_texts,
			padding=True,
			truncation=True,
			max_length=512,
			return_tensors="pt",
		).to(device)
		with torch.no_grad():
			outputs = model.generate(
				input_ids=input_encodings["input_ids"],
				attention_mask=input_encodings["attention_mask"],
				output_scores=True,
				return_dict_in_generate=True,
			)

		# Extract logits for the 'false' and 'true' tokens from the model's output
		logits = outputs.scores[-1][:, [token_false_id, token_true_id]]
		# Calculate the softmax probability of the 'true' token
		probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
		results.extend(probs.tolist())
	return results
