import asyncio
import os
from typing import List, Tuple, Optional

import cohere
from cohere import RerankResponseResultsItem

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import process_batch


@passage_reranker_node
def cohere_reranker(queries: List[str], contents_list: List[List[str]],
                    scores_list: List[List[float]], ids_list: List[List[str]],
                    top_k: int, batch: int = 64,
                    model: str = "rerank-multilingual-v2.0",
                    api_key: Optional[str] = None) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Rerank a list of contents with Cohere rerank models.
    You can get the API key from https://cohere.com/rerank and set it in the environment variable COHERE_API_KEY.

    :param queries: The list of queries to use for reranking
    :param contents_list: The list of lists of contents to rerank
    :param scores_list: The list of lists of scores retrieved from the initial ranking
    :param ids_list: The list of lists of ids retrieved from the initial ranking
    :param top_k: The number of passages to be retrieved
    :param batch: The number of queries to be processed in a batch
    :param model: The model name for Cohere rerank.
        You can choose between "rerank-multilingual-v2.0" and "rerank-english-v2.0".
        Default is "rerank-multilingual-v2.0".
    :param api_key: The API key for Cohere rerank.
        You can set it in the environment variable COHERE_API_KEY.
        Or, you can directly set it on the config YAML file using this parameter.
        Default is env variable "COHERE_API_KEY".
    :return: Tuple of lists containing the reranked contents, ids, and scores
    """
    api_key = os.getenv("COHERE_API_KEY", None) if api_key is None else api_key
    if api_key is None:
        raise KeyError("Please set the API key for Cohere rerank in the environment variable COHERE_API_KEY "
                       "or directly set it on the config YAML file.")

    cohere_client = cohere.AsyncClient(api_key)

    # Run async cohere_rerank_pure function
    tasks = [cohere_rerank_pure(cohere_client, model, query, document, ids, top_k) for query, document, ids in
             zip(queries, contents_list, ids_list)]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_batch(tasks, batch_size=batch))
    content_result = list(map(lambda x: x[0], results))
    id_result = list(map(lambda x: x[1], results))
    score_result = list(map(lambda x: x[2], results))

    del cohere_client

    return content_result, id_result, score_result


async def cohere_rerank_pure(cohere_client: cohere.AsyncClient, model: str,
                             query: str, documents: List[str], ids: List[str],
                             top_k: int) -> Tuple[List[str], List[str], List[float]]:
    """
    Rerank a list of contents with Cohere rerank models.

    :param cohere_client: The Cohere AsyncClient to use for reranking
    :param model: The model name for Cohere rerank
    :param query: The query to use for reranking
    :param documents: The list of contents to rerank
    :param ids: The list of ids corresponding to the documents
    :param top_k: The number of passages to be retrieved
    :return: Tuple of lists containing the reranked contents, ids, and scores
    """
    rerank_results = await cohere_client.rerank(model=model, query=query, documents=documents, top_n=top_k,
                                                return_documents=False)
    results: List[RerankResponseResultsItem] = rerank_results.results
    reranked_scores: List[float] = list(map(lambda x: x.relevance_score, results))
    indices = list(map(lambda x: x.index, results))
    reranked_contents: List[str] = list(map(lambda i: documents[i], indices))
    reranked_ids: List[str] = list(map(lambda i: ids[i], indices))
    return reranked_contents, reranked_ids, reranked_scores
