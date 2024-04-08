import asyncio
from typing import List, Tuple

import torch
from sentence_transformers import CrossEncoder

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import process_batch


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossEncoder(
        model_name, max_length=sentence_transformer_max_length, device=device
    )
    tasks = [sentence_transformer_reranker_pure(query, contents, scores, top_k, ids, model)
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


async def sentence_transformer_reranker_pure(query: str, contents: List[str], scores: List[float], top_k: int,
                                             ids: List[str], model) -> Tuple[List[str], List[str], List[float]]:
    """
    Rerank a list of contents based on their relevance to a query using Sentence Transformer model.

    :param query: The query to use for reranking
    :param contents: The list of contents to rerank
    :param scores: The list of scores retrieved from the initial ranking
    :param ids: The list of ids retrieved from the initial ranking
    :param top_k: The number of passages to be retrieved
    :param model: The name of the Sentence Transformer model to use for reranking
    :return: tuple of lists containing the reranked contents, ids, and scores
    """
    input_texts = [(query, content) for content in contents]
    with torch.no_grad():
        pred_scores = model.predict(sentences=input_texts, apply_softmax=True)

    content_ids_probs = list(zip(contents, ids, pred_scores.tolist()))

    # Sort the list of pairs based on the relevance score in descending order
    sorted_content_ids_probs = sorted(content_ids_probs, key=lambda x: x[2], reverse=True)

    # crop with top_k
    if len(contents) < top_k:
        top_k = len(contents)
    sorted_content_ids_probs = sorted_content_ids_probs[:top_k]

    content_result, id_result, score_result = zip(*sorted_content_ids_probs)

    return list(content_result), list(id_result), list(score_result)
