import functools
import os
import pickle
from pathlib import Path
from typing import List, Union, Tuple, Dict

import chromadb
import pandas as pd

from autorag import embedding_models
from autorag.strategy import select_best_average
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
            # check if chroma_path and file exists
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
        elif func.__name__ == "hybrid_rrf":
            target_modules = kwargs.pop("target_modules")
            assert isinstance(target_modules, tuple), "target_modules must be tuple."
            assert len(target_modules) > 1, "target_modules must have at least 2 modules."
            node_dir = kwargs.pop("node_dir")
            assert node_dir is not None, "You must pass node_dir for using hybrid retrieval."
            target_ids, target_scores = get_evaluation_result(node_dir, target_modules)
            ids, scores = func(target_ids, target_scores, **kwargs)
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


def get_evaluation_result(node_dir: str, target_modules: Tuple) -> Tuple[Tuple, Tuple]:
    """
    Get ids and scores of target_module from summary.parquet and each result parquet file.

    :param node_dir: The directory of the node.
    :param target_modules: The name of the target modules.
    :return: A tuple of ids and tuple of scores at each target module.
    """
    def select_best_among_module(df: pd.DataFrame, module_name: str):
        modules_summary = df.loc[lambda row: row['module_name'] == module_name]
        if len(modules_summary) == 1:
            return modules_summary.iloc[0, :]
        elif len(modules_summary) <= 0:
            raise ValueError(f"module_name {module_name} does not exist in summary.parquet. "
                             f"You must run {module_name} before running hybrid retrieval.")
        metrics = modules_summary.drop(columns=['filename', 'module_name', 'module_params', 'execution_time'])
        metric_average = metrics.mean(axis=1)
        metric_average = metric_average.reset_index(drop=True)
        max_idx = metric_average.idxmax()
        best_module = modules_summary.iloc[max_idx, :]
        return best_module

    summary_df = pd.read_parquet(os.path.join(node_dir, "summary.parquet"))
    best_results = list(map(lambda module_name: select_best_among_module(summary_df, module_name), target_modules))
    best_results_df = list(map(lambda df: pd.read_parquet(os.path.join(node_dir, df['filename'])), best_results))
    ids = tuple(map(lambda df: df['retrieved_ids'].apply(list).tolist(), best_results_df))
    scores = tuple(map(lambda df: df['retrieve_scores'].apply(list).tolist(), best_results_df))
    return ids, scores
