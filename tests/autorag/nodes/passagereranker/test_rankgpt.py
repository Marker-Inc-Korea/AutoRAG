from unittest.mock import patch

import pytest
from llama_index.core import QueryBundle
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.openai import OpenAI

from autorag.nodes.passagereranker import RankGPT
from autorag.nodes.passagereranker.rankgpt import AsyncRankGPTRerank
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	queries_example,
	contents_example,
	scores_example,
	ids_example,
	base_reranker_test,
	project_dir,
	previous_result,
	base_reranker_node_test,
)


async def mock_openai_achat(self, messages) -> ChatResponse:
	return ChatResponse(
		message=ChatMessage(content="[2] > [1] > [3]", role=MessageRole.ASSISTANT)
	)


@pytest.fixture
def rankgpt_instance():
	return RankGPT(project_dir, "openai", model="gpt-4o-mini")


@patch.object(
	OpenAI,
	"achat",
	mock_openai_achat,
)
@pytest.mark.asyncio()
async def test_async_rankgpt_rerank():
	query = queries_example[0]
	query_bundle = QueryBundle(query_str=query)
	nodes = list(
		map(lambda x: NodeWithScore(node=TextNode(text=x)), contents_example[0])
	)

	reranker = AsyncRankGPTRerank(top_n=3, llm=OpenAI())
	result, id_result = await reranker.async_postprocess_nodes(nodes, query_bundle)

	assert len(result) == 3
	assert all(isinstance(node, NodeWithScore) for node in result)


@patch.object(
	OpenAI,
	"achat",
	mock_openai_achat,
)
def test_rankgpt_reranker(rankgpt_instance):
	top_k = 3
	contents_result, id_result, score_result = rankgpt_instance._pure(
		queries_example,
		contents_example,
		scores_example,
		ids_example,
		top_k,
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(
	OpenAI,
	"achat",
	mock_openai_achat,
)
def test_rankgpt_reranker_batch_one(rankgpt_instance):
	top_k = 3
	batch = 1
	contents_result, id_result, score_result = rankgpt_instance._pure(
		queries_example,
		contents_example,
		scores_example,
		ids_example,
		top_k,
		batch=batch,
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(
	OpenAI,
	"achat",
	mock_openai_achat,
)
def test_rankgpt_node():
	top_k = 1
	result_df = RankGPT.run_evaluator(
		project_dir=project_dir,
		previous_result=previous_result,
		top_k=top_k,
		llm="openai",
		model="gpt-3.5-turbo",
		temperature=0.5,
		batch=8,
		verbose=True,
	)
	base_reranker_node_test(result_df, top_k)

	top_k = 2
	result_df = RankGPT.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, top_k=top_k, batch=4
	)
	base_reranker_node_test(result_df, top_k)

	result_df = RankGPT.run_evaluator(
		project_dir=project_dir,
		previous_result=previous_result,
		top_k=top_k,
		batch=4,
		llm=OpenAI(model="gpt-4o"),
	)
	base_reranker_node_test(result_df, top_k)
