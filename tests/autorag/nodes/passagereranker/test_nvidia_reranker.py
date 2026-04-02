import aiohttp
import pytest
from aioresponses import aioresponses

from autorag.nodes.passagereranker import NvidiaReranker
from autorag.nodes.passagereranker.nvidia import nvidia_rerank_pure
from autorag.utils.util import get_event_loop
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

NVIDIA_RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
MOCK_RESPONSE = {
	"rankings": [
        {"index": 1, "logit": 0.9},
        {"index": 0, "logit": 0.2},
        {"index": 2, "logit": 0.1},
    ]
}

@pytest.fixture
def nvidia_reranker_instance():
	reranker = NvidiaReranker(project_dir=project_dir, api_key="test")
	yield reranker
	if hasattr(reranker, "session") and not reranker.session.closed:
		loop = get_event_loop()
		if loop.is_running():
			loop.create_task(reranker.session.close())
		else:
			loop.run_until_complete(reranker.session.close())

@pytest.mark.asyncio()
async def test_nvidia_rerank_pure():
	with aioresponses() as m:
		m.post(NVIDIA_RERANK_URL, payload=MOCK_RESPONSE)

		async with aiohttp.ClientSession() as session:
			session.headers.update(
				{"Authorization": "Bearer mock_api_key", "Accept": "application/json"}
			)

			documents = ["doc0", "doc1", "doc2"]
			ids = ["id0", "id1", "id2"]

			content_result, id_result, score_result = await nvidia_rerank_pure(
				session,
				NVIDIA_RERANK_URL,
				"nvidia/rerank-qa-mistral-4b",
				queries_example[0],
				documents,
				ids,
				top_k=2,
			)
		
		assert len(content_result) == 2
		assert len(id_result) == 2
		assert len(score_result) == 2

		assert all([res in documents for res in content_result])
		assert all([res in ids for res in id_result])

		assert score_result[0] >= score_result[1]


@pytest.mark.asyncio()
async def test_nvidia_rerank_pure_raises_when_rankings_length_mismatch():
	with aioresponses() as m:
		mock_response = {"rankings": [{"index": 0, "logit": 0.9}]}
		m.post(NVIDIA_RERANK_URL, payload=mock_response)

		async with aiohttp.ClientSession() as session:
			session.headers.update(
				{"Authorization": "Bearer mock_api_key", "Accept": "application/json"}
			)
			with pytest.raises(AssertionError):
				await nvidia_rerank_pure(
					session,
					NVIDIA_RERANK_URL,
					"nvidia/rerank-qa-mistral-4b",
					queries_example[0],
					["doc0", "doc1"],
					["id0", "id1"],
					top_k=2,
				)

def test_nvidia_reranker(nvidia_reranker_instance):
    with aioresponses() as m:
        m.post(NVIDIA_RERANK_URL, payload=MOCK_RESPONSE, repeat=True)
        
        top_k = 3
        contents_result, id_result, score_result = nvidia_reranker_instance._pure(
            queries_example, contents_example, scores_example, ids_example, top_k
        )
        base_reranker_test(contents_result, id_result, score_result, top_k)


def test_nvidia_reranker_batch_one(nvidia_reranker_instance):
    with aioresponses() as m:
        m.post(NVIDIA_RERANK_URL, payload=MOCK_RESPONSE, repeat=True)
        
        top_k = 3
        batch = 1
        contents_result, id_result, score_result = nvidia_reranker_instance._pure(
            queries_example,
            contents_example,
            scores_example,
            ids_example,
            top_k,
            batch=batch,
        )
        base_reranker_test(contents_result, id_result, score_result, top_k)


def test_nvidia_reranker_node():
    with aioresponses() as m:
        m.post(NVIDIA_RERANK_URL, payload=MOCK_RESPONSE, repeat=True)
        
        top_k = 1
        result_df = NvidiaReranker.run_evaluator(
            project_dir=project_dir,
            previous_result=previous_result,
            top_k=top_k,
            api_key="test",
        )
        base_reranker_node_test(result_df, top_k)
