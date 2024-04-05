import asyncio
import os

import aiohttp

from autorag.nodes.passagereranker.jina import jina_reranker_pure
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import queries_example, contents_example, \
    scores_example, ids_example


def test_jina_reranker_pure():
    api_key = os.environ['JINAAI_API_KEY']
    assert bool(api_key) is True

    session = aiohttp.ClientSession()
    session.headers.update({"Authorization": f"Bearer {api_key}", "Accept-Encoding": "identity"})
    content_result, id_result, score_result = asyncio.run(
        jina_reranker_pure(queries_example[0], contents_example[0], scores_example[0], ids_example[0],
                           session, top_k=2, api_key=api_key))
    assert len(content_result) == 2
    assert len(id_result) == 2
    assert len(score_result) == 2

    assert all([res in contents_example[0] for res in content_result])
    assert all([res in ids_example[0] for res in id_result])

    # check if the scores are sorted
    assert score_result[0] >= score_result[1]

    session.close()
