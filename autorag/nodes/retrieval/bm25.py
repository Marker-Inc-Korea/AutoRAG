import asyncio
import pickle
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

from autorag.nodes.retrieval.base import retrieval_node, evenly_distribute_passages
from autorag.utils import validate_corpus_dataset


@retrieval_node
def bm25(queries: List[List[str]], top_k: int, bm25_corpus: Dict) -> Tuple[List[List[str
]], List[List[float]]]:
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
                "tokens": [], # 2d list of tokens
                "passage_id": [], # 2d list of passage_id.
            }

    :return: The 2-d list contains a list of passage ids that retrieved from bm25 and 2-d list of its scores.
        It will be a length of queries. And each element has a length of top_k.
    """
    # check if bm25_corpus is valid
    assert ("tokens" and "passage_id" in list(bm25_corpus.keys())), \
        "bm25_corpus must contain tokens and passage_id. Please check you ingested bm25 corpus correctly."
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    bm25_instance = BM25Okapi(bm25_corpus["tokens"])
    # run async bm25_pure function
    tasks = [bm25_pure(input_queries, top_k, tokenizer, bm25_instance, bm25_corpus) for input_queries in queries]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    id_result = list(map(lambda x: x[0], results))
    score_result = list(map(lambda x: x[1], results))
    return id_result, score_result


async def bm25_pure(queries: List[str], top_k: int, tokenizer, bm25_api: BM25Okapi, bm25_corpus: Dict) -> Tuple[
        List[str], List[float]]:
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
                "tokens": [], # 2d list of tokens
                "passage_id": [], # 2d list of passage_id. Type must be str.
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
    # sort id_result and score_result by score
    result = [(_id, score) for score, _id in
              sorted(zip(score_result, id_result), key=lambda pair: pair[0], reverse=True)]
    id_result, score_result = zip(*result)
    return list(id_result), list(score_result)


def bm25_ingest(corpus_path: str, corpus_data: pd.DataFrame):
    if not corpus_path.endswith('.pkl'):
        raise ValueError(f"Corpus path {corpus_path} is not a pickle file.")
    validate_corpus_dataset(corpus_data)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False)
    tasks = list(
        map(lambda x: bm25_tokenize(x[0], x[1], tokenizer), zip(corpus_data['contents'], corpus_data['doc_id'])))
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    tokenized_corpus, passage_ids = zip(*results)
    bm25_dict = {
        'tokens': list(tokenized_corpus),
        'passage_id': list(passage_ids),
    }
    with open(corpus_path, 'wb') as w:
        pickle.dump(bm25_dict, w)


async def bm25_tokenize(queries: List[str], passage_id: str, tokenizer) -> Tuple[List[int], str]:
    tokenized_queries = tokenizer(queries).input_ids
    return tokenized_queries, passage_id
