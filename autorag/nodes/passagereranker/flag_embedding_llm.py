from typing import List, Tuple

import torch
from FlagEmbedding import FlagLLMReranker

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.nodes.passagereranker.flag_embedding import flag_embedding_run_model
from autorag.utils.util import sort_and_select_top_k, flatten_apply


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
    nested_list = [list(map(lambda x: [query, x], content_list)) for query, content_list in zip(queries, contents_list)]
    rerank_scores = flatten_apply(flag_embedding_run_model, nested_list, model=model, batch_size=batch)
    sorted_contents, sorted_ids, sorted_scores = sort_and_select_top_k(contents_list, ids_list, rerank_scores, top_k)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sorted_contents, sorted_ids, sorted_scores
