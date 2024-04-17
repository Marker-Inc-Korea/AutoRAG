from typing import List, Tuple

import pandas as pd
import torch
from sentence_transformers import CrossEncoder

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import flatten_apply, make_batch, select_top_k, sort_by_scores


@passage_reranker_node
def sentence_transformer_reranker(queries: List[str], contents_list: List[List[str]],
                                  scores_list: List[List[float]], ids_list: List[List[str]],
                                  top_k: int, batch: int = 64, sentence_transformer_max_length: int = 512,
                                  model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
                                  ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Rerank a list of contents based on their relevance to a query using Sentence Transformer model.

    :param queries: The list of queries to use for reranking
    :param contents_list: The list of lists of contents to rerank
    :param scores_list: The list of lists of scores retrieved from the initial ranking
    :param ids_list: The list of lists of ids retrieved from the initial ranking
    :param top_k: The number of passages to be retrieved
    :param batch: The number of queries to be processed in a batch
    :param sentence_transformer_max_length: The maximum length of the input text for the Sentence Transformer model
    :param model_name: The name of the Sentence Transformer model to use for reranking
        Default is "cross-encoder/ms-marco-MiniLM-L-2-v2"
    :return: tuple of lists containing the reranked contents, ids, and scores
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(
        model_name, max_length=sentence_transformer_max_length, device=device
    )

    nested_list = [list(map(lambda x: [query, x], content_list)) for query, content_list in zip(queries, contents_list)]
    rerank_scores = flatten_apply(sentence_transformer_run_model, nested_list, model=model, batch_size=batch)

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


def sentence_transformer_run_model(input_texts, model, batch_size: int):
    batch_input_texts = make_batch(input_texts, batch_size)
    results = []
    for batch_texts in batch_input_texts:
        with torch.no_grad():
            pred_scores = model.predict(sentences=batch_texts, apply_softmax=True)
        results.extend(pred_scores.tolist())
    return results
