import asyncio

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.openai import OpenAI

from autorag.nodes.passagereranker import rankgpt
from autorag.nodes.passagereranker.rankgpt import AsyncRankGPTRerank
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import queries_example, contents_example, \
    scores_example, ids_example, base_reranker_test, project_dir, previous_result, base_reranker_node_test


def test_async_rankgpt_rerank():
    query = queries_example[0]
    query_bundle = QueryBundle(query_str=query)
    nodes = list(map(lambda x: NodeWithScore(node=TextNode(text=x)), contents_example[0]))

    reranker = AsyncRankGPTRerank(top_n=3, llm=OpenAI())
    result, id_result = asyncio.run(reranker.async_postprocess_nodes(nodes, query_bundle))

    assert len(result) == 3
    assert all(isinstance(node, NodeWithScore) for node in result)


def test_rankgpt_reranker():
    top_k = 3
    original_rankgpt_reranker = rankgpt.__wrapped__
    contents_result, id_result, score_result \
        = original_rankgpt_reranker(queries_example, contents_example, scores_example, ids_example, top_k,
                                    llm=OpenAI(model="gpt-3.5-turbo-16k"))
    base_reranker_test(contents_result, id_result, score_result, top_k)


def test_rankgpt_reranker_batch_one():
    top_k = 3
    batch = 1
    original_rankgpt_reranker = rankgpt.__wrapped__
    contents_result, id_result, score_result \
        = original_rankgpt_reranker(queries_example, contents_example, scores_example, ids_example, top_k,
                                    llm=OpenAI(model="gpt-3.5-turbo-16k"), batch=batch)
    base_reranker_test(contents_result, id_result, score_result, top_k)


def test_rankgpt_node():
    top_k = 1
    result_df = rankgpt(project_dir=project_dir, previous_result=previous_result, top_k=top_k,
                        llm='openai', model='gpt-3.5-turbo', temperature=0.5, batch=8, verbose=True)
    base_reranker_node_test(result_df, top_k)

    top_k = 2
    result_df = rankgpt(project_dir=project_dir, previous_result=previous_result, top_k=top_k, batch=4)
    base_reranker_node_test(result_df, top_k)
