import asyncio
import os

from autorag.nodes.passagereranker import jina_reranker
from autorag.nodes.passagereranker.jina import jina_reranker_pure
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import queries_example, contents_example, \
    scores_example, ids_example, base_reranker_test, project_dir, previous_result, base_reranker_node_test


def test_jina_reranker_pure():
    api_key = os.environ['JINAAI_API_KEY']
    assert bool(api_key) is True
    content_result, id_result, score_result = asyncio.run(
        jina_reranker_pure(queries_example[0], contents_example[0], scores_example[0], ids_example[0], top_k=2,
                           api_key=api_key))
    assert len(content_result) == 2
    assert len(id_result) == 2
    assert len(score_result) == 2

    assert all([res in contents_example[0] for res in content_result])
    assert all([res in ids_example[0] for res in id_result])

    # check if the scores are sorted
    assert score_result[0] >= score_result[1]


def test_jina_reranker():
    top_k = 3
    original_jina_reranker = jina_reranker.__wrapped__
    contents_result, id_result, score_result \
        = original_jina_reranker(queries_example, contents_example, scores_example, ids_example, top_k)
    base_reranker_test(contents_result, id_result, score_result, top_k)


def test_jina_reranker_batch_one():
    top_k = 3
    batch = 1
    original_jina_reranker = jina_reranker.__wrapped__
    contents_result, id_result, score_result \
        = original_jina_reranker(queries_example, contents_example, scores_example, ids_example, top_k, batch=batch)
    base_reranker_test(contents_result, id_result, score_result, top_k)


def test_jina_reranker_node():
    top_k = 1
    result_df = jina_reranker(project_dir=project_dir, previous_result=previous_result, top_k=top_k)
    base_reranker_node_test(result_df, top_k)
