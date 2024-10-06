from unittest.mock import patch

import pytest

import autorag
from autorag.nodes.passagereranker import MixedbreadAIReranker
from autorag.nodes.passagereranker.mixedbreadai import mixedbreadai_rerank_pure
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	queries_example,
	contents_example,
	ids_example,
	base_reranker_test,
	project_dir,
	previous_result,
	base_reranker_node_test,
)


async def mock_mixedbreadai_reranker(client, query, documents, ids, top_k, **kwargs):
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
def mixedbreadai_reranker_instance():
	return MixedbreadAIReranker(project_dir=project_dir, api_key="mock_api_key")


@patch.object(
	autorag.nodes.passagereranker.mixedbreadai,
	"mixedbreadai_rerank_pure",
	mock_mixedbreadai_reranker,
)
def test_mixedbreadai_reranker(mixedbreadai_reranker_instance):
	top_k = 1
	contents_result, id_result, score_result = mixedbreadai_reranker_instance._pure(
		queries_example, contents_example, ids_example, top_k
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(
	autorag.nodes.passagereranker.mixedbreadai,
	"mixedbreadai_rerank_pure",
	mock_mixedbreadai_reranker,
)
def test_mixedbreadai_reranker_batch_one(mixedbreadai_reranker_instance):
	top_k = 1
	batch = 1
	contents_result, id_result, score_result = mixedbreadai_reranker_instance._pure(
		queries_example, contents_example, ids_example, top_k, batch=batch
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(
	autorag.nodes.passagereranker.mixedbreadai,
	"mixedbreadai_rerank_pure",
	mock_mixedbreadai_reranker,
)
def test_mixedbreadai_node():
	top_k = 1
	result_df = MixedbreadAIReranker.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, top_k=top_k
	)
	base_reranker_node_test(result_df, top_k)
