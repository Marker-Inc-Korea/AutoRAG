import asyncio
from typing import List, Union, Dict, Tuple

from uuid import UUID

import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer


# decorator method that cast queries to list, get contents from db


def cast(queries: Union[str, List[str]]) -> List[str]:
    if isinstance(queries, str):
        return [queries]
    elif isinstance(queries, List):
        return queries
    else:
        raise ValueError(f"queries must be str or list, but got {type(queries)}")


def evenly_distribute_passages(ids: List[List[UUID]], scores: List[List[float]], top_k: int) -> Tuple[List[UUID], List[float]]:
    assert len(ids) == len(scores), "ids and scores must have same length."
    query_cnt = len(ids)
    avg_len = top_k // query_cnt
    remainder = top_k % query_cnt

    new_ids = []
    new_scores = []
    for i in range(query_cnt):
        if i < remainder:
            new_ids.extend(ids[i][:avg_len + 1])
            new_scores.extend(scores[i][:avg_len + 1])
        else:
            new_ids.extend(ids[i][:avg_len])
            new_scores.extend(scores[i][:avg_len])

    return new_ids, new_scores


def bm25(queries: List[List[str]], top_k: int, bm25_corpus: Dict) -> List[Tuple[List[UUID], List[float]]]:
    """
    BM25 retrieval function.
    You have to load a pickle file that is already ingested.

    :param queries: 2-d list of query strings.
    Each element of the list is a query strings of each row.
    :param top_k: The number of passages to be retrieved.
    :param bm25_corpus: A dictionary containing the bm25 corpus, which is doc_id from corpus and tokenized corpus.
    Its data structure looks like this:

    .. Code:: python

        {
            "Tokens": [], # 2d list of tokens
            "passage_id": [], # 2d list of passage_id. Type must be UUID.
        }

    :return: The List of tuple contains a list of passage ids that retrieved from bm25 and its scores.
    It will be a length of queries. And each element has a length of top_k.
    """
    # check if bm25_corpus is valid
    assert ("tokens" and "passage_id" in list(bm25_corpus.keys())), \
        "bm25_corpus must contain tokens and passage_id. Please check you ingested bm25 corpus correctly."
    # tokenize input query
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    bm25_instance = BM25Okapi(bm25_corpus["tokens"])



async def bm25_pure(queries: List[str], top_k: int, tokenizer, bm25_api: BM25Okapi, bm25_corpus: Dict) -> Tuple[
    List[UUID], List[float]]:
    """
    Async BM25 retrieval function.
    Its usage is for async retrieval of bm25 row by row.

    :param queries: A list of query strings.
    :param top_k: The number of passages to be retrieved.
    :param tokenizer: A tokenizer that will be used to tokenize queries.
    :param bm25_api: A bm25 api instance that will be used to retrieve passages.
    :param bm25_corpus: A dictionary containing the bm25 corpus, which is doc_id from corpus and tokenized corpus.
    Its data structure looks like this:

    .. Code:: python

        {
            "Tokens": [], # 2d list of tokens
            "passage_id": [], # 2d list of passage_id. Type must be UUID.
        }
    :return: The tuple contains a list of passage ids that retrieved from bm25 and its scores.
    """
    # I don't make queries operation to async, because queries length might be small, so it will occur overhead.
    tokenized_queries = tokenizer(queries).input_ids
    id_result = []
    score_result = []
    for query in tokenized_queries:
        scores = bm25_api.get_scores(query)
        sorted_scores = sorted(scores, reverse=True)
        top_n_index = np.argsort(scores)[::-1][:top_k]
        ids = [bm25_corpus['passage_id'][i] for i in top_n_index]
        id_result.append(ids)
        score_result.append(sorted_scores[:top_k])

    # make a total result to top_k
    id_result, score_result = evenly_distribute_passages(id_result, score_result, top_k)
    return id_result, score_result
