from unittest.mock import patch

import cohere.base_client
from cohere import RerankResponse, RerankResponseResultsItem

from autorag.nodes.passagereranker import cohere_reranker
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import queries_example, contents_example, \
    scores_example, ids_example, base_reranker_test, project_dir, previous_result, base_reranker_node_test


async def mock_cohere_reranker(self, model, query, documents, top_n, **kwargs) -> RerankResponse:
    if query == queries_example[0]:
        return RerankResponse(
            results=[
                        RerankResponseResultsItem(index=1, relevance_score=0.8),
                        RerankResponseResultsItem(index=2, relevance_score=0.2),
                        RerankResponseResultsItem(index=0, relevance_score=0.1),
                    ][:top_n]
        )
    elif query == queries_example[1]:
        return RerankResponse(
            results=[
                        RerankResponseResultsItem(index=2, relevance_score=0.8),
                        RerankResponseResultsItem(index=0, relevance_score=0.2),
                        RerankResponseResultsItem(index=1, relevance_score=0.1),
                    ][:top_n]
        )
    response_items = [
        RerankResponseResultsItem(index=i, relevance_score=0.1 * i)
        for i in range(len(documents) - 1, -1, -1)
    ]
    return RerankResponse(results=response_items[:top_n])


@patch.object(cohere.base_client.AsyncBaseCohere, "rerank", mock_cohere_reranker)
def test_cohere_reranker():
    top_k = 3
    original_cohere_reranker = cohere_reranker.__wrapped__
    contents_result, id_result, score_result \
        = original_cohere_reranker(queries_example, contents_example, scores_example, ids_example, top_k)
    base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(cohere.base_client.AsyncBaseCohere, "rerank", mock_cohere_reranker)
def test_cohere_reranker_batch_one():
    top_k = 3
    batch = 1
    original_cohere_reranker = cohere_reranker.__wrapped__
    contents_result, id_result, score_result \
        = original_cohere_reranker(queries_example, contents_example, scores_example, ids_example, top_k, batch=batch)
    base_reranker_test(contents_result, id_result, score_result, top_k)


@patch.object(cohere.base_client.AsyncBaseCohere, "rerank", mock_cohere_reranker)
def test_cohere_node():
    top_k = 1
    result_df = cohere_reranker(project_dir=project_dir, previous_result=previous_result, top_k=top_k)
    base_reranker_node_test(result_df, top_k)
