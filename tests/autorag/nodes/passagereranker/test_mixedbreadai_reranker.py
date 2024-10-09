from unittest.mock import patch

import pytest

import mixedbread_ai.client
from mixedbread_ai.types.reranking_response import RerankingResponse
from mixedbread_ai.types.ranked_document import RankedDocument
from mixedbread_ai.types.usage import Usage

from autorag.nodes.passagereranker import MixedbreadAIReranker
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	queries_example,
	contents_example,
	ids_example,
	base_reranker_test,
	project_dir,
	previous_result,
	base_reranker_node_test,
)


async def mock_mixedbreadai_reranker(
	self,
	*,
	query,
	input,
	model,
	top_k,
	**kwargs,
):
	mock_usage = Usage(prompt_tokens=100, total_tokens=150, completion_tokens=50)
	mock_documents = [
		RankedDocument(index=1, score=0.8, input="Document 1", object=None),
		RankedDocument(index=2, score=0.2, input="Document 2", object=None),
		RankedDocument(index=0, score=0.1, input="Document 3", object=None),
	]
	return RerankingResponse(
		usage=mock_usage,
		model="mock-model",
		data=mock_documents[:top_k],
		object=None,
		top_k=top_k,
		return_input=False,
	)


@pytest.fixture
def mixedbreadai_reranker_instance():
	return MixedbreadAIReranker(project_dir=project_dir, api_key="mock_api_key")


@patch.object(
	mixedbread_ai.client.AsyncMixedbreadAI, "reranking", mock_mixedbreadai_reranker
)
def test_mixedbreadai_reranker(mixedbreadai_reranker_instance):
	top_k = 1
	contents_result, id_result, score_result = mixedbreadai_reranker_instance._pure(
		queries_example, contents_example, ids_example, top_k
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(
	mixedbread_ai.client.AsyncMixedbreadAI, "reranking", mock_mixedbreadai_reranker
)
def test_mixedbreadai_reranker_batch_one(mixedbreadai_reranker_instance):
	top_k = 1
	batch = 1
	contents_result, id_result, score_result = mixedbreadai_reranker_instance._pure(
		queries_example, contents_example, ids_example, top_k, batch=batch
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(
	mixedbread_ai.client.AsyncMixedbreadAI, "reranking", mock_mixedbreadai_reranker
)
def test_mixedbreadai_node():
	top_k = 1
	result_df = MixedbreadAIReranker.run_evaluator(
		project_dir=project_dir,
		previous_result=previous_result,
		top_k=top_k,
		api_key="mock",
	)
	base_reranker_node_test(result_df, top_k)
