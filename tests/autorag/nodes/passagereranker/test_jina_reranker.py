from unittest.mock import patch

import aiohttp
import pytest
from aioresponses import aioresponses

import autorag
from autorag.nodes.passagereranker import JinaReranker
from autorag.nodes.passagereranker.jina import jina_reranker_pure, JINA_API_URL
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	queries_example,
	contents_example,
	ids_example,
	base_reranker_test,
	project_dir,
	previous_result,
	base_reranker_node_test,
)


@pytest.mark.asyncio()
async def test_jina_reranker_pure():
	with aioresponses() as m:
		mock_response = {
			"results": [
				{"index": 1, "relevance_score": 0.9},
				{"index": 0, "relevance_score": 0.2},
			]
		}
		m.post(JINA_API_URL, payload=mock_response)
		session = aiohttp.ClientSession()
		session.headers.update(
			{"Authorization": "Bearer mock_api_key", "Accept-Encoding": "identity"}
		)
		content_result, id_result, score_result = await jina_reranker_pure(
			session,
			queries_example[0],
			contents_example[0],
			ids_example[0],
			top_k=2,
		)
		assert len(content_result) == 2
		assert len(id_result) == 2
		assert len(score_result) == 2

		assert all([res in contents_example[0] for res in content_result])
		assert all([res in ids_example[0] for res in id_result])

		# check if the scores are sorted
		assert score_result[0] >= score_result[1]


async def mock_jina_reranker_pure(session, query, contents, ids, top_k, **kwargs):
	if query == queries_example[0]:
		return (
			[contents[1], contents[2], contents[0]][:top_k],
			[ids[1], ids[2], ids[0]][:top_k],
			[0.8, 0.2, 0.1][:top_k],
		)
	elif query == queries_example[1]:
		return (
			[contents[1], contents[0], contents[2]][:top_k],
			[ids[1], ids[0], ids[2]][:top_k],
			[0.8, 0.2, 0.1][:top_k],
		)
	else:
		raise ValueError(f"Unexpected query: {query}")


@pytest.fixture
def jina_reranker_instance():
	return JinaReranker(project_dir, "mock_api_key")


@patch.object(
	autorag.nodes.passagereranker.jina, "jina_reranker_pure", mock_jina_reranker_pure
)
def test_jina_reranker(jina_reranker_instance):
	top_k = 3
	contents_result, id_result, score_result = jina_reranker_instance._pure(
		queries_example, contents_example, ids_example, top_k
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(
	autorag.nodes.passagereranker.jina, "jina_reranker_pure", mock_jina_reranker_pure
)
def test_jina_reranker_batch_one(jina_reranker_instance):
	top_k = 3
	batch = 1
	contents_result, id_result, score_result = jina_reranker_instance._pure(
		queries_example,
		contents_example,
		ids_example,
		top_k,
		batch=batch,
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(
	autorag.nodes.passagereranker.jina, "jina_reranker_pure", mock_jina_reranker_pure
)
def test_jina_reranker_node():
	top_k = 1
	result_df = JinaReranker.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, top_k=top_k
	)
	base_reranker_node_test(result_df, top_k)
