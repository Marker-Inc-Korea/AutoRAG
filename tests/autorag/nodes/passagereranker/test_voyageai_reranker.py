from unittest.mock import patch

import pytest

from collections import namedtuple
import voyageai
from voyageai.object.reranking import RerankingObject, RerankingResult
from voyageai.api_resources import VoyageResponse

import autorag
from autorag.nodes.passagereranker import VoyageAIReranker
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	queries_example,
	contents_example,
	ids_example,
	base_reranker_test,
	project_dir,
	previous_result,
	base_reranker_node_test,
)


async def mock_voyageai_reranker_pure(
	self,
	query,
	documents,
	model,
	top_k,
	truncation,
):
	mock_documents = ["Document 1 content", "Document 2 content", "Document 3 content"]

	# Mock response data
	mock_response_data = [
		{"index": 1, "relevance_score": 0.8},
		{"index": 2, "relevance_score": 0.2},
		{"index": 0, "relevance_score": 0.1},
	]

	# Mock usage data
	mock_usage = {"total_tokens": 100}

	# Create a mock VoyageResponse object
	mock_response = VoyageResponse()
	mock_response.data = [
		namedtuple("MockData", d.keys())(*d.values()) for d in mock_response_data
	]
	mock_response.usage = namedtuple("MockUsage", mock_usage.keys())(
		*mock_usage.values()
	)

	# Create an instance of RerankingObject using the mock data
	object = RerankingObject(documents=mock_documents, response=mock_response)

	if top_k == 1:
		object.results = [
			RerankingResult(index=1, document="nodonggunn", relevance_score=0.8)
		]
	return object


@pytest.fixture
def voyageai_reranker_instance():
	return VoyageAIReranker(project_dir, api_key="mock_api_key")


@patch.object(voyageai.AsyncClient, "rerank", mock_voyageai_reranker_pure)
def test_voyageai_reranker(voyageai_reranker_instance):
	top_k = 3
	contents_result, id_result, score_result = voyageai_reranker_instance._pure(
		queries_example, contents_example, ids_example, top_k
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(voyageai.AsyncClient, "rerank", mock_voyageai_reranker_pure)
def test_voyageai_reranker_batch_one(voyageai_reranker_instance):
	top_k = 1
	batch = 1
	contents_result, id_result, score_result = voyageai_reranker_instance._pure(
		queries_example, contents_example, ids_example, top_k, batch=batch
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(voyageai.AsyncClient, "rerank", mock_voyageai_reranker_pure)
def test_voyageai_reranker_node():
	top_k = 1
	result_df = VoyageAIReranker.run_evaluator(
		project_dir=project_dir,
		previous_result=previous_result,
		top_k=top_k,
		api_key="mock_api_key",
	)
	base_reranker_node_test(result_df, top_k)
