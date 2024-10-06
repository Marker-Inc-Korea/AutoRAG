from unittest.mock import patch

import pytest

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


def mock_voyageai_reranker_pure(
	voyage_client, model, query, documents, ids, top_k, truncation
):
	if query == queries_example[0]:
		return (
			[documents[1], documents[2], documents[0]][:top_k],
			[ids[1], ids[2], ids[0]][:top_k],
			[0.8, 0.2, 0.1][:top_k],
		)
	elif query == queries_example[1]:
		return (
			[documents[1], documents[0], documents[2]][:top_k],
			[ids[1], ids[0], ids[2]][:top_k],
			[0.8, 0.2, 0.1][:top_k],
		)
	else:
		raise ValueError(f"Unexpected query: {query}")


@pytest.fixture
def voyageai_reranker_instance():
	return VoyageAIReranker(project_dir, api_key="mock_api_key")


@patch.object(
	autorag.nodes.passagereranker.voyageai,
	"voyageai_rerank_pure",
	mock_voyageai_reranker_pure,
)
def test_voyageai_reranker(voyageai_reranker_instance):
	top_k = 3
	contents_result, id_result, score_result = voyageai_reranker_instance._pure(
		queries_example, contents_example, ids_example, top_k
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(
	autorag.nodes.passagereranker.voyageai,
	"voyageai_rerank_pure",
	mock_voyageai_reranker_pure,
)
def test_voyageai_reranker_node():
	top_k = 1
	result_df = VoyageAIReranker.run_evaluator(
		project_dir=project_dir,
		previous_result=previous_result,
		top_k=top_k,
		api_key="mock_api_key",
	)
	base_reranker_node_test(result_df, top_k)
