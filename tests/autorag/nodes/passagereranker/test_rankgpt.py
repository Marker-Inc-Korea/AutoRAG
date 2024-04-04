import asyncio

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.openai import OpenAI

from autorag.nodes.passagereranker.rankgpt import AsyncRankGPTRerank
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import queries_example, contents_example


def test_async_rankgpt_rerank():
    query = queries_example[0]
    query_bundle = QueryBundle(query_str=query)
    nodes = list(map(lambda x: NodeWithScore(node=TextNode(text=x)), contents_example[0]))

    reranker = AsyncRankGPTRerank(top_n=3, llm=OpenAI())
    result = asyncio.run(reranker.async_postprocess_nodes(nodes, query_bundle))

    assert len(result) == 3
    assert all(isinstance(node, NodeWithScore) for node in result)
