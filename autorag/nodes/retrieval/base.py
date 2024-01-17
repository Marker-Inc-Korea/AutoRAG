import functools
import os
import pickle
from pathlib import Path
from typing import List, Union, Tuple, Dict

import pandas as pd

from autorag.utils import fetch_contents, result_to_dataframe, validate_qa_dataset


def retrieval_node(func):
    """
    Load resources for running retrieval_node.
    For example, it loads bm25 corpus for bm25 retrieval.

    :param func: Retrieval function that returns a list of ids and a list of scores
    :return: A pandas Dataframe that contains retrieved contents, retrieved ids, and retrieve scores.
        The column name will be "retrieved_contents", "retrieved_ids", and "retrieve_scores".
    """

    @functools.wraps(func)
    @result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
    def wrapper(
            project_dir: Union[str, Path],
            previous_result: pd.DataFrame,
            *args, **kwargs) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        validate_qa_dataset(previous_result)
        resources_dir = os.path.join(project_dir, "resources")
        data_dir = os.path.join(project_dir, "data")
        bm25_path = os.path.join(resources_dir, 'bm25.pkl')
        if func.__name__ == "bm25":
            assert bm25_path is not None, "bm25_path must be specified for using bm25 retrieval."
            assert os.path.exists(bm25_path), f"bm25_path {bm25_path} does not exist. Please ingest first."
        # TODO: add chroma check for vectordb

        # find queries columns & type cast queries
        assert "query" in previous_result.columns, "previous_result must have query column."
        if "queries" not in previous_result.columns:
            previous_result["queries"] = previous_result["query"]
        previous_result["queries"] = previous_result["queries"].apply(cast_queries)
        queries = previous_result["queries"].tolist()

        bm25_corpus = load_bm25_corpus(bm25_path)

        # run retrieval function
        if func.__name__ == "bm25":
            ids, scores = func(queries=queries, bm25_corpus=bm25_corpus, *args, **kwargs)
        else:
            raise ValueError(f"invalid func name for using retrieval_io decorator.")
        # TODO: add chroma load for vectordb

        # fetch data from corpus_data
        corpus_data = pd.read_parquet(os.path.join(data_dir, "corpus.parquet"))
        contents = fetch_contents(corpus_data, ids)

        return contents, ids, scores

    return wrapper


def load_bm25_corpus(bm25_path: str) -> Dict:
    if bm25_path is None:
        return {}
    with open(bm25_path, "rb") as f:
        bm25_corpus = pickle.load(f)
    return bm25_corpus


def cast_queries(queries: Union[str, List[str]]) -> List[str]:
    if isinstance(queries, str):
        return [queries]
    elif isinstance(queries, List):
        return queries
    else:
        raise ValueError(f"queries must be str or list, but got {type(queries)}")


def evenly_distribute_passages(ids: List[List[str]], scores: List[List[float]], top_k: int) -> Tuple[
    List[str], List[float]]:
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
