from typing import List, Optional, Sequence

from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.utils import print_text

from autorag.nodes.passagereranker.base import passage_reranker_node


@passage_reranker_node
def rankgpt(queries: List[str], contents_list: List[List[str]],
            scores_list: List[List[float]], ids_list: List[List[str]],
            top_k: int, llm: Optional[LLM] = None):
    pass


class AsyncRankGPTRerank(RankGPTRerank):
    async def async_run_llm(self, messages: Sequence[ChatMessage]) -> ChatResponse:
        return await self.llm.achat(messages)

    async def async_postprocess_nodes(self,
                                      nodes: List[NodeWithScore],
                                      query_bundle: QueryBundle,
                                      ) -> List[NodeWithScore]:
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

            for idx in rerank_ranks:
                initial_results.append(
                    NodeWithScore(node=nodes[idx].node, score=nodes[idx].score)
                )
            return initial_results[: self.top_n]
        else:
            return nodes[: self.top_n]
