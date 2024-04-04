import asyncio
from typing import List, Optional, Sequence, Tuple

import numpy as np
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.utils import print_text
from llama_index.llms.openai import OpenAI

from autorag import generator_models
from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import process_batch


@passage_reranker_node
def rankgpt(queries: List[str], contents_list: List[List[str]],
            scores_list: List[List[float]], ids_list: List[List[str]],
            top_k: int, llm: Optional[LLM] = None, verbose: bool = False,
            rankgpt_rerank_prompt: Optional[str] = None,
            batch: int = 16, **kwargs) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Rerank given context paragraphs using RankGPT.
    Return pseudo scores, since the actual scores are not available on RankGPT.

    :param queries: The list of queries to use for reranking
    :param contents_list: The list of lists of contents to rerank
    :param scores_list: The list of lists of scores retrieved from the initial ranking
    :param ids_list: The list of lists of ids retrieved from the initial ranking
    :param top_k: The number of passages to be retrieved
    :param llm: The LLM model to use for RankGPT rerank.
        It is LlamaIndex model.
        Default is OpenAI model with gpt-3.5-turbo-16k.
    :param verbose: Whether to print intermediate steps.
    :param rankgpt_rerank_prompt: The prompt template for RankGPT rerank.
        Default is RankGPT's default prompt.
    :param batch: The number of queries to be processed in a batch.
    :return: Tuple of lists containing the reranked contents, ids, and scores
    """
    query_bundles = list(map(lambda query: QueryBundle(query_str=query), queries))
    nodes_list = [
        list(map(lambda x: NodeWithScore(node=TextNode(text=x[0]), score=x[1]), zip(content_list, score_list)))
        for content_list, score_list in zip(contents_list, scores_list)]
    if llm is None:
        llm = OpenAI(model="gpt-3.5-turbo-16k")

    if not isinstance(llm, LLM):
        llm = generator_models[llm](**kwargs)

    reranker = AsyncRankGPTRerank(top_n=top_k, llm=llm, verbose=verbose, rankgpt_rerank_prompt=rankgpt_rerank_prompt)

    tasks = [reranker.async_postprocess_nodes(nodes, query, ids) for nodes, query, ids in
             zip(nodes_list, query_bundles, ids_list)]
    loop = asyncio.get_event_loop()
    rerank_result = loop.run_until_complete(process_batch(tasks, batch_size=batch))
    content_result = [list(map(lambda x: x.node.text, res[0])) for res in rerank_result]
    score_result = [np.linspace(1.0, 0.0, len(res[0])).tolist() for res in rerank_result]
    id_result = [res[1] for res in rerank_result]

    del reranker
    del llm

    return content_result, id_result, score_result


class AsyncRankGPTRerank(RankGPTRerank):
    async def async_run_llm(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        return await self.llm.achat(messages)

    async def async_postprocess_nodes(self,
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
