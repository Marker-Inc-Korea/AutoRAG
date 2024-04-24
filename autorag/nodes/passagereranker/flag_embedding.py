from typing import List, Tuple

import pandas as pd
import torch
from FlagEmbedding import FlagReranker

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import make_batch, sort_by_scores, flatten_apply, select_top_k


@passage_reranker_node
def flag_embedding_reranker(queries: List[str], contents_list: List[List[str]],
                            scores_list: List[List[float]], ids_list: List[List[str]],
                            top_k: int, batch: int = 64, use_fp16: bool = False,
                            model_name: str = "BAAI/bge-reranker-large",
                            ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Rerank a list of contents based on their relevance to a query using BAAI normal-Reranker model.

    :param queries: The list of queries to use for reranking
    :param contents_list: The list of lists of contents to rerank
    :param scores_list: The list of lists of scores retrieved from the initial ranking
    :param ids_list: The list of lists of ids retrieved from the initial ranking
    :param top_k: The number of passages to be retrieved
    :param batch: The number of queries to be processed in a batch
        Default is 64.
    :param use_fp16: Whether to use fp16 for inference
    :param model_name: The name of the BAAI Reranker normal-model name.
        Default is "BAAI/bge-reranker-large"
    :return: tuple of lists containing the reranked contents, ids, and scores
    """

    model = FlagReranker(
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


def flag_embedding_run_model(input_texts, model, batch_size: int):
    batch_input_texts = make_batch(input_texts, batch_size)
    results = []
    for batch_texts in batch_input_texts:
        with torch.no_grad():
            pred_scores = model.compute_score(sentence_pairs=batch_texts)
        if batch_size == 1:
            results.append(pred_scores)
        else:
            results.extend(pred_scores)
    return results
