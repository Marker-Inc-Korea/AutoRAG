import logging
from typing import List, Tuple

import pandas as pd

from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils import result_to_dataframe
from autorag.utils.util import select_top_k, sort_by_scores, empty_cuda_cache

logger = logging.getLogger("AutoRAG")


class Upr(BasePassageReranker):
	def __init__(
		self,
		project_dir: str,
		use_bf16: bool = False,
		prefix_prompt: str = "Passage: ",
		suffix_prompt: str = "Please write a question based on this passage.",
		*args,
		**kwargs,
	):
		"""
		Initialize the UPR reranker node.

		:param project_dir: The project directory
		:param use_bf16: Whether to use bfloat16 for the model. Default is False.
		:param prefix_prompt: The prefix prompt for the language model that generates question for reranking.
			Default is "Passage: ".
			The prefix prompt serves as the initial context or instruction for the language model.
			It sets the stage for what is expected in the output
		:param suffix_prompt: The suffix prompt for the language model that generates question for reranking.
			Default is "Please write a question based on this passage.".
			The suffix prompt provides a cue or a closing instruction to the language model,
				signaling how to conclude the generated text or what format to follow at the end.
		:param kwargs: Extra arguments
		"""
		super().__init__(project_dir, *args, **kwargs)

		self.scorer = UPRScorer(
			suffix_prompt=suffix_prompt, prefix_prompt=prefix_prompt, use_bf16=use_bf16
		)

	def __del__(self):
		del self.scorer
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, _, ids = self.cast_to_run(previous_result)
		top_k = kwargs.pop("top_k")
		return self._pure(queries, contents, ids, top_k)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		ids_list: List[List[str]],
		top_k: int,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank a list of contents based on their relevance to a query using UPR.
		UPR is a reranker based on UPR (https://github.com/DevSinghSachan/unsupervised-passage-reranking).
		The language model will make a question based on the passage and rerank the passages by the likelihood of the question.
		The default model is t5-large.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved

		:return: tuple of lists containing the reranked contents, ids, and scores
		"""
		df = pd.DataFrame(
			{
				"query": queries,
				"contents": contents_list,
				"ids": ids_list,
			}
		)

		df["scores"] = df.apply(
			lambda row: self.scorer.compute(
				query=row["query"], contents=row["contents"]
			),
			axis=1,
		)
		df[["contents", "ids", "scores"]] = df.apply(
			lambda x: sort_by_scores(x, reverse=False), axis=1, result_type="expand"
		)
		results = select_top_k(df, ["contents", "ids", "scores"], top_k)
		return (
			results["contents"].tolist(),
			results["ids"].tolist(),
			results["scores"].tolist(),
		)


class UPRScorer:
	def __init__(self, suffix_prompt: str, prefix_prompt: str, use_bf16: bool = False):
		try:
			import torch
			from transformers import T5Tokenizer, T5ForConditionalGeneration
		except ImportError:
			raise ImportError(
				"torch is not installed. Please install torch to use UPRReranker."
			)
		model_name = "t5-large"
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.tokenizer = T5Tokenizer.from_pretrained(model_name)
		self.model = T5ForConditionalGeneration.from_pretrained(
			model_name, torch_dtype=torch.bfloat16 if use_bf16 else torch.float32
		).to(self.device)
		self.suffix_prompt = suffix_prompt
		self.prefix_prompt = prefix_prompt

	def compute(self, query: str, contents: List[str]) -> List[float]:
		try:
			import torch
		except ImportError:
			raise ImportError(
				"torch is not installed. Please install torch to use UPRReranker."
			)
		query_token = self.tokenizer(
			query, max_length=128, truncation=True, return_tensors="pt"
		)
		prompts = list(
			map(
				lambda content: f"{self.prefix_prompt} {content} {self.suffix_prompt}",
				contents,
			)
		)
		prompt_token_outputs = self.tokenizer(
			prompts,
			padding="longest",
			max_length=512,
			pad_to_multiple_of=8,
			truncation=True,
			return_tensors="pt",
		)

		query_input_ids = torch.repeat_interleave(
			query_token["input_ids"], len(contents), dim=0
		).to(self.device)

		with torch.no_grad():
			logits = self.model(
				input_ids=prompt_token_outputs["input_ids"].to(self.device),
				attention_mask=prompt_token_outputs["attention_mask"].to(self.device),
				labels=query_input_ids,
			).logits
		log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
		nll = -log_softmax.gather(2, query_input_ids.unsqueeze(2)).squeeze(2)
		avg_nll = torch.sum(nll, dim=1)
		return avg_nll.tolist()

	def __del__(self):
		del self.model
		del self.tokenizer
		empty_cuda_cache()
