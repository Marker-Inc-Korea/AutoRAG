import functools
import os
import pickle
from pathlib import Path
from typing import List, Union, Tuple, Dict

import chromadb
import pandas as pd

from autorag import embedding_models
from autorag.support import get_support_modules
from autorag.utils import fetch_contents, result_to_dataframe, validate_qa_dataset

import logging

logger = logging.getLogger("AutoRAG")


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
            **kwargs) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        validate_qa_dataset(previous_result)
        resources_dir = os.path.join(project_dir, "resources")
        data_dir = os.path.join(project_dir, "data")

        if func.__name__ == "bm25":
            # check if bm25_path and file exists
            bm25_path = os.path.join(resources_dir, 'bm25.pkl')
            assert bm25_path is not None, "bm25_path must be specified for using bm25 retrieval."
            assert os.path.exists(bm25_path), f"bm25_path {bm25_path} does not exist. Please ingest first."
        elif func.__name__ == "vectordb":
            # check if chroma_path and file exist
            chroma_path = os.path.join(resources_dir, 'chroma')
            embedding_model_str = kwargs.pop("embedding_model")
            assert chroma_path is not None, "chroma_path must be specified for using vectordb retrieval."
            assert os.path.exists(chroma_path), f"chroma_path {chroma_path} does not exist. Please ingest first."

        # find queries columns & type cast queries
        assert "query" in previous_result.columns, "previous_result must have query column."
        if "queries" not in previous_result.columns:
            previous_result["queries"] = previous_result["query"]
        previous_result["queries"] = previous_result["queries"].apply(cast_queries)
        queries = previous_result["queries"].tolist()

        # run retrieval function
        if func.__name__ == "bm25":
            bm25_corpus = load_bm25_corpus(bm25_path)
            ids, scores = func(queries=queries, bm25_corpus=bm25_corpus, **kwargs)
        elif func.__name__ == "vectordb":
            chroma_collection = load_chroma_collection(db_path=chroma_path, collection_name=embedding_model_str)
            if embedding_model_str in embedding_models:
                embedding_model = embedding_models[embedding_model_str]
            else:
                logger.error(f"embedding_model_str {embedding_model_str} does not exist.")
                raise KeyError(f"embedding_model_str {embedding_model_str} does not exist.")
            ids, scores = func(queries=queries, collection=chroma_collection,
                               embedding_model=embedding_model, **kwargs)
        elif func.__name__ in ["hybrid_rrf", "hybrid_cc"]:
            if 'ids' in kwargs and 'scores' in kwargs:
                ids, scores = func(**kwargs)
            else:
                if not ('target_modules' in kwargs and 'target_module_params' in kwargs):
                    raise ValueError(
                        f"If there are no ids and scores specified, target_modules and target_module_params must be specified for using {func.__name__}.")
                target_modules = kwargs.pop('target_modules')
                target_module_params = kwargs.pop('target_module_params')
                result_dfs = list(map(lambda x: get_support_modules(x[0])(**x[1], project_dir=project_dir,
                                                                          previous_result=previous_result),
                                      zip(target_modules, target_module_params)))
                ids = tuple(map(lambda df: df['retrieved_ids'].apply(list).tolist(), result_dfs))
                scores = tuple(map(lambda df: df['retrieve_scores'].apply(list).tolist(), result_dfs))
                ids, scores = func(ids=ids, scores=scores, **kwargs)
        else:
            raise ValueError(f"invalid func name for using retrieval_io decorator.")

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


def load_chroma_collection(db_path: str, collection_name: str) -> chromadb.Collection:
    db = chromadb.PersistentClient(path=db_path)
    collection = db.get_collection(name=collection_name)
    return collection


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


def run_retrieval_modules(project_dir: str, previous_result: pd.DataFrame,
                          module_name: str, module_params: Dict) -> pd.DataFrame:
    return
