from typing import List, Tuple

import pandas as pd
import torch
from FlagEmbedding import FlagLLMReranker

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.nodes.passagereranker.flag_embedding import flag_embedding_run_model
from autorag.utils.util import flatten_apply, sort_by_scores, select_top_k


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
        Default is 64.
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

    df = pd.DataFrame({
        'contents': contents_list,
        'ids': ids_list,
        'scores': rerank_scores,
    })
    df[['contents', 'ids', 'scores']] = df.apply(sort_by_scores, axis=1, result_type='expand')
    results = select_top_k(df, ['contents', 'ids', 'scores'], top_k)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results['contents'].tolist(), results['ids'].tolist(), results['scores'].tolist()
