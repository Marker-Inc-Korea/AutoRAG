from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.utils import print_text
from llama_index.llms.openai import OpenAI

from autorag import generator_models
from autorag.nodes.passagereranker.base import BasePassageReranker
from autorag.utils.util import (
	get_event_loop,
	process_batch,
	pop_params,
	result_to_dataframe,
	empty_cuda_cache,
)


class RankGPT(BasePassageReranker):
	def __init__(
		self, project_dir: str, llm: Optional[Union[str, LLM]] = None, **kwargs
	):
		"""
		Initialize the RankGPT reranker.

		:param project_dir: The project directory
		:param llm: The LLM model to use for RankGPT rerank.
			It is a llama index model.
			Default is the OpenAI model with gpt-4o-mini.
		:param kwargs: The keyword arguments for the LLM model.
		"""
		super().__init__(project_dir)
		if llm is None:
			self.llm = OpenAI(model="gpt-4o-mini")
		else:
			if not isinstance(llm, LLM):
				llm_class = generator_models[llm]
				llm_param = pop_params(llm_class.__init__, kwargs)
				self.llm = llm_class(**llm_param)
			else:
				self.llm = llm

	def __del__(self):
		del self.llm
		empty_cuda_cache()
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, contents, scores, ids = self.cast_to_run(previous_result)
		top_k = kwargs.get("top_k", 1)
		verbose = kwargs.get("verbose", False)
		rankgpt_rerank_prompt = kwargs.get("rankgpt_rerank_prompt", None)
		batch = kwargs.get("batch", 16)
		return self._pure(
			queries=queries,
			contents_list=contents,
			scores_list=scores,
			ids_list=ids,
			top_k=top_k,
			verbose=verbose,
			rankgpt_rerank_prompt=rankgpt_rerank_prompt,
			batch=batch,
		)

	def _pure(
		self,
		queries: List[str],
		contents_list: List[List[str]],
		scores_list: List[List[float]],
		ids_list: List[List[str]],
		top_k: int,
		verbose: bool = False,
		rankgpt_rerank_prompt: Optional[str] = None,
		batch: int = 16,
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		"""
		Rerank given context paragraphs using RankGPT.
		Return pseudo scores, since the actual scores are not available on RankGPT.

		:param queries: The list of queries to use for reranking
		:param contents_list: The list of lists of contents to rerank
		:param scores_list: The list of lists of scores retrieved from the initial ranking
		:param ids_list: The list of lists of ids retrieved from the initial ranking
		:param top_k: The number of passages to be retrieved
		:param verbose: Whether to print intermediate steps.
		:param rankgpt_rerank_prompt: The prompt template for RankGPT rerank.
			Default is RankGPT's default prompt.
		:param batch: The number of queries to be processed in a batch.
		:return: Tuple of lists containing the reranked contents, ids, and scores
		"""
		query_bundles = list(map(lambda query: QueryBundle(query_str=query), queries))
		nodes_list = [
			list(
				map(
					lambda x: NodeWithScore(node=TextNode(text=x[0]), score=x[1]),
					zip(content_list, score_list),
				)
			)
			for content_list, score_list in zip(contents_list, scores_list)
		]

		reranker = AsyncRankGPTRerank(
			top_n=top_k,
			llm=self.llm,
			verbose=verbose,
			rankgpt_rerank_prompt=rankgpt_rerank_prompt,
		)

		tasks = [
			reranker.async_postprocess_nodes(nodes, query, ids)
			for nodes, query, ids in zip(nodes_list, query_bundles, ids_list)
		]
		loop = get_event_loop()
		rerank_result = loop.run_until_complete(process_batch(tasks, batch_size=batch))
		content_result = [
			list(map(lambda x: x.node.text, res[0])) for res in rerank_result
		]
		score_result = [
			np.linspace(1.0, 0.0, len(res[0])).tolist() for res in rerank_result
		]
		id_result = [res[1] for res in rerank_result]

		del reranker

		return content_result, id_result, score_result


class AsyncRankGPTRerank(RankGPTRerank):
	async def async_run_llm(self, messages: Sequence[ChatMessage]) -> ChatResponse:
		return await self.llm.achat(messages)

	async def async_postprocess_nodes(
		self,
		nodes: List[NodeWithScore],
		query_bundle: QueryBundle,
		ids: Optional[List[str]] = None,
	) -> Tuple[List[NodeWithScore], List[str]]:
		if ids is None:
			ids = [str(i) for i in range(len(nodes))]

		items = {
			"query": query_bundle.query_str,
			"hits": [{"content": node.get_content()} for node in nodes],
		}

		messages = self.create_permutation_instruction(item=items)
		permutation = await self.async_run_llm(messages=messages)
		if permutation.message is not None and permutation.message.content is not None:
			rerank_ranks = self._receive_permutation(
				items, str(permutation.message.content)
			)
			if self.verbose:
				print_text(f"After Reranking, new rank list for nodes: {rerank_ranks}")

			initial_results: List[NodeWithScore] = []
			id_results = []

			for idx in rerank_ranks:
				initial_results.append(
					NodeWithScore(node=nodes[idx].node, score=nodes[idx].score)
				)
				id_results.append(ids[idx])
			return initial_results[: self.top_n], id_results[: self.top_n]
		else:
			return nodes[: self.top_n], ids[: self.top_n]
