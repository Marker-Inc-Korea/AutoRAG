import asyncio
from typing import List, Tuple

import torch
from FlagEmbedding import FlagLLMReranker

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.nodes.passagereranker.flag_embedding import flag_embedding_reranker_pure
from autorag.utils.util import process_batch


@passage_reranker_node
def flag_embedding_llm_reranker(queries: List[str], contents_list: List[List[str]],
                                scores_list: List[List[float]], ids_list: List[List[str]],
                                top_k: int, batch: int = 64, use_fp16: bool = False,
                                model_name: str = "BAAI/bge-reranker-v2-gemma",
                                ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Rerank a list of contents based on their relevance to a query using BAAI LLM-based-Reranker model.

    :param queries: The list of queries to use for reranking
    :param contents_list: The list of lists of contents to rerank
    :param scores_list: The list of lists of scores retrieved from the initial ranking
    :param ids_list: The list of lists of ids retrieved from the initial ranking
    :param top_k: The number of passages to be retrieved
    :param batch: The number of queries to be processed in a batch
    :param use_fp16: Whether to use fp16 for inference
    :param model_name: The name of the BAAI Reranker LLM-based-model name.
        Default is "BAAI/bge-reranker-v2-gemma"
    :return: tuple of lists containing the reranked contents, ids, and scores
    """
    model = FlagLLMReranker(
        model_name_or_path=model_name, use_fp16=use_fp16
    )
    tasks = [flag_embedding_reranker_pure(query, contents, scores, top_k, ids, model)
             for query, contents, scores, ids in zip(queries, contents_list, scores_list, ids_list)]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_batch(tasks, batch_size=batch))
    content_result = list(map(lambda x: x[0], results))
    id_result = list(map(lambda x: x[1], results))
    score_result = list(map(lambda x: x[2], results))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return content_result, id_result, score_result
