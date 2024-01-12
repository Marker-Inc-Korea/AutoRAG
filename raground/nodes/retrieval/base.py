import functools
import os
import pathlib
import pickle
from copy import deepcopy
from pathlib import Path
from typing import List, Union, Tuple
from uuid import UUID

import pandas as pd

from raground.utils import validate_corpus_dataset


def retrieval_node(func):
    @functools.wraps(func)
    def wrapper(
            node_line_dir: Path,
            corpus_dataset: pd.DataFrame,
            previous_result: pd.DataFrame,
            *args, **kwargs):
        resources_dir = os.path.join(pathlib.PurePath(node_line_dir).parent.parent, "resources")
        bm25_path = os.path.join(resources_dir, 'bm25.pkl')
        if func.__name__ == "bm25":
            assert bm25_path is not None, "bm25_path must be specified for using bm25 retrieval."
            assert os.path.exists(bm25_path), f"bm25_path {bm25_path} does not exist. Please ingest first."
        # TODO: add chroma check for vectordb
        validate_corpus_dataset(corpus_dataset)

        assert "queries" in previous_result.columns, "previous_result must have queries column."
        assert "query" in previous_result.columns, "previous_result must have query column."
        previous_result["queries"] = previous_result["queries"].apply(cast)
        queries = previous_result["queries"].tolist()

        # TODO: make final evaluate func with evaluate_retrieval decorator

        # load bm25 corpus
        bm25_corpus = None
        if bm25_path is not None:
            with open(bm25_path, "rb") as f:
                bm25_corpus = pickle.load(f)

        if func.__name__ == "bm25":
            result = func(queries, bm25_corpus, *args, **kwargs)
        else:
            raise ValueError(f"invalid func name for using retrieval_io decorator.")
        # TODO: add chroma load for vectordb

        # record result to csv files, return result as pd.DataFrame
        result_df = deepcopy(previous_result)
        result_df = result_df.drop('queries', axis=1)
        result_df = result_df.concat([result_df,
                                      pd.DataFrame(result, columns=['ids', 'scores'])], axis=1)

    return wrapper


def cast(queries: Union[str, List[str]]) -> List[str]:
    if isinstance(queries, str):
        return [queries]
    elif isinstance(queries, List):
        return queries
    else:
        raise ValueError(f"queries must be str or list, but got {type(queries)}")


def evenly_distribute_passages(ids: List[List[UUID]], scores: List[List[float]], top_k: int) -> Tuple[
    List[UUID], List[float]]:
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


