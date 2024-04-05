from typing import List, Tuple, Optional

import aiohttp

from autorag.nodes.passagereranker.base import passage_reranker_node

JINA_API_URL = "https://api.jina.ai/v1/rerank"


@passage_reranker_node
def jina_reranker(queries: List[str], contents_list: List[List[str]],
                  scores_list: List[List[float]], ids_list: List[List[str]],
                  top_k: int, api_key: Optional[str] = None,
                  model: str = "jina-reranker-v1-base-en",
                  batch: int = 8
                  ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    session = aiohttp.ClientSession()
    session.headers.update(
        {"Authorization": f"Bearer {api_key}", "Accept-Encoding": "identity"}
    )
    return


async def jina_reranker_pure(query: str, contents: List[str],
                             scores: List[float], ids: List[str],
                             request_session: aiohttp.ClientSession,
                             top_k: int, api_key: str,
                             model: str = "jina-reranker-v1-base-en") -> Tuple[List[str], List[str], List[float]]:
    async with request_session.post(
            JINA_API_URL,
            json={
                "query": query,
                "documents": contents,
                "model": model,
                "top_n": top_k,
            },
    ) as resp:
        resp_json = await resp.json()
        if 'results' not in resp_json:
            raise RuntimeError(f"Invalid response from Jina API: {resp_json['detail']}")

        results = resp_json['results']
        indices = list(map(lambda x: x['index'], results))
        score_result = list(map(lambda x: x['relevance_score'], results))
        id_result = list(map(lambda x: ids[x], indices))
        content_result = list(map(lambda x: contents[x], indices))

        return content_result, id_result, score_result
