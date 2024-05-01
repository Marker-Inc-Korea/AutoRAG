import asyncio
import os
import pickle
import re
from typing import List, Dict, Tuple, Callable, Union, Iterable

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi, Token
from llama_index.core.indices.keyword_table.utils import simple_extract_keywords
from nltk import PorterStemmer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from autorag.nodes.retrieval.base import retrieval_node, evenly_distribute_passages
from autorag.utils import validate_corpus_dataset
from autorag.utils.util import normalize_string


def tokenize_ko_kiwi(texts: List[str]) -> List[List[str]]:
    texts = list(map(lambda x: x.strip().lower(), texts))
    kiwi = Kiwi()
    tokenized_list: Iterable[List[Token]] = kiwi.tokenize(texts)
    return [list(map(lambda x: x.form, token_list)) for token_list in tokenized_list]


def tokenize_porter_stemmer(texts: List[str]) -> List[List[str]]:
    def tokenize_remove_stopword(text: str, stemmer) -> List[str]:
        text = text.lower()
        words = list(simple_extract_keywords(text))
        return [stemmer.stem(word) for word in words]

    stemmer = PorterStemmer()
    tokenized_list: List[List[str]] = list(map(lambda x: tokenize_remove_stopword(x, stemmer), texts))
    return tokenized_list


def tokenize_space(texts: List[str]) -> List[List[str]]:
    def tokenize_space_text(text: str) -> List[str]:
        text = normalize_string(text)
        return re.split(r'\s+', text.strip())

    return list(map(tokenize_space_text, texts))


BM25_TOKENIZER = {
    'porter_stemmer': tokenize_porter_stemmer,
    'ko_kiwi': tokenize_ko_kiwi,
    'space': tokenize_space,
}


@retrieval_node
def bm25(queries: List[List[str]], top_k: int, bm25_corpus: Dict, bm25_tokenizer: str = 'porter_stemmer') -> \
        Tuple[List[List[str]], List[List[float]]]:
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

    :param bm25_tokenizer: The tokenizer name that uses to the BM25.
        It supports 'porter_stemmer', 'ko_kiwi', and huggingface `AutoTokenizer`.
        You can pass huggingface tokenizer name.
        Default is porter_stemmer.
    :return: The 2-d list contains a list of passage ids that retrieved from bm25 and 2-d list of its scores.
        It will be a length of queries. And each element has a length of top_k.
    """
    # check if bm25_corpus is valid
    assert ("tokens" and "passage_id" in list(bm25_corpus.keys())), \
        "bm25_corpus must contain tokens and passage_id. Please check you ingested bm25 corpus correctly."
    tokenizer = select_bm25_tokenizer(bm25_tokenizer)
    assert bm25_corpus['tokenizer_name'] == bm25_tokenizer, \
        (f"The bm25 corpus tokenizer is {bm25_corpus['tokenizer_name']}, but your input is {bm25_tokenizer}. "
         f"You need to ingest again. Delete bm25 pkl file and re-ingest it.")
    bm25_instance = BM25Okapi(bm25_corpus["tokens"])
    # run async bm25_pure function
    tasks = [bm25_pure(input_queries, top_k, tokenizer, bm25_instance, bm25_corpus) for input_queries in queries]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    id_result = list(map(lambda x: x[0], results))
    score_result = list(map(lambda x: x[1], results))
    return id_result, score_result


async def bm25_pure(queries: List[str], top_k: int, tokenizer, bm25_api: BM25Okapi, bm25_corpus: Dict) -> \
        Tuple[List[str], List[float]]:
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
    if isinstance(tokenizer, PreTrainedTokenizerBase):
        tokenized_queries = tokenizer(queries).input_ids
    else:
        tokenized_queries = tokenizer(queries)
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


def bm25_ingest(corpus_path: str, corpus_data: pd.DataFrame, bm25_tokenizer: str = 'porter_stemmer'):
    if not corpus_path.endswith('.pkl'):
        raise ValueError(f"Corpus path {corpus_path} is not a pickle file.")
    validate_corpus_dataset(corpus_data)
    ids = corpus_data['doc_id'].tolist()

    # Initialize bm25_corpus
    bm25_corpus = pd.DataFrame()

    # Load the BM25 corpus if it exists and get the passage ids
    if os.path.exists(corpus_path) and os.path.getsize(corpus_path) > 0:
        with open(corpus_path, 'rb') as r:
            corpus = pickle.load(r)
            bm25_corpus = pd.DataFrame.from_dict(corpus)
        duplicated_passage_rows = bm25_corpus[bm25_corpus['passage_id'].isin(ids)]
        new_passage = corpus_data[~corpus_data['doc_id'].isin(duplicated_passage_rows['passage_id'])]
    else:
        new_passage = corpus_data

    if not new_passage.empty:
        tokenizer = select_bm25_tokenizer(bm25_tokenizer)
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            tokenized_corpus = tokenizer(new_passage['contents'].tolist()).input_ids
        else:
            tokenized_corpus = tokenizer(new_passage['contents'].tolist())
        new_bm25_corpus = pd.DataFrame({
            'tokens': tokenized_corpus,
            'passage_id': new_passage['doc_id'].tolist(),
        })

        if not bm25_corpus.empty:
            bm25_corpus_updated = pd.concat([bm25_corpus, new_bm25_corpus], ignore_index=True)
            bm25_dict = bm25_corpus_updated.to_dict('list')
        else:
            bm25_dict = new_bm25_corpus.to_dict('list')

        # add tokenizer name to bm25_dict
        bm25_dict['tokenizer_name'] = bm25_tokenizer

        with open(corpus_path, 'wb') as w:
            pickle.dump(bm25_dict, w)


def select_bm25_tokenizer(bm25_tokenizer: str) -> Callable[[str], List[Union[int, str]]]:
    if bm25_tokenizer in list(BM25_TOKENIZER.keys()):
        return BM25_TOKENIZER[bm25_tokenizer]

    return AutoTokenizer.from_pretrained(bm25_tokenizer, use_fast=False)
