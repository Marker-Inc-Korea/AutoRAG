import functools
import os
from pathlib import Path
from typing import List, Union, Tuple

import pandas as pd

from autorag import embedding_models
from autorag.nodes.retrieval import bm25, vectordb
from autorag.nodes.retrieval.base import load_bm25_corpus, load_chroma_collection
from autorag.utils import fetch_contents, result_to_dataframe, validate_qa_dataset

import logging

logger = logging.getLogger("AutoRAG")


def query_expansion_node(func):
    @functools.wraps(func)
    @result_to_dataframe(["expanded_queries", "retrieved_contents", "retrieved_ids", "retrieve_scores"])
    def wrapper(
            project_dir: Union[str, Path],
            previous_result: pd.DataFrame,
            *args, **kwargs) -> Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[float]]]:
        validate_qa_dataset(previous_result)
        resources_dir = os.path.join(project_dir, "resources")
        data_dir = os.path.join(project_dir, "data")

        # find queries columns & type cast queries
        assert "query" in previous_result.columns, "previous_result must have query column."
        queries = previous_result["query"].tolist()

        # run query expansion function
        if func.__name__ == "query_decompose":
            decomposed_queries = func(queries=queries, *args, **kwargs)
        elif func.__name__ == "hyde":
            decomposed_queries = func(queries=queries, *args, **kwargs)
            pass
        else:
            raise ValueError(f"Unknown query expansion function: {func.__name__}")

        # get retrieval module values
        if "retrieval_module" not in kwargs.keys():
            retrieval_module = "bm25"  # default retrieval module is bm25
        else:
            retrieval_module = kwargs.pop("retrieval_module")

        # set parameters for retrieval module
        if retrieval_module == "bm25":
            # check if bm25_path and file exists
            bm25_path = os.path.join(resources_dir, "resources", 'bm25.pkl')
            assert bm25_path is not None, "bm25_path must be specified for using bm25 retrieval."
            assert os.path.exists(bm25_path), f"bm25_path {bm25_path} does not exist. Please ingest first."
        elif retrieval_module == "vectordb":
            # check if chroma_path and file exists
            chroma_path = os.path.join(resources_dir, 'chroma')
            if "embedding_model" not in kwargs.keys():
                embedding_model_str = "openai"  # default embedding model is openai
            else:
                embedding_model_str = kwargs.pop("embedding_model")
            assert chroma_path is not None, "chroma_path must be specified for using vectordb retrieval."
            assert os.path.exists(chroma_path), f"chroma_path {chroma_path} does not exist. Please ingest first."

        # run retrieval function
        if retrieval_module == "bm25":
            bm25_corpus = load_bm25_corpus(bm25_path)
            ids, scores = bm25(queries=decomposed_queries, bm25_corpus=bm25_corpus, *args, **kwargs)
        elif retrieval_module == "vectordb":
            chroma_collection = load_chroma_collection(db_path=chroma_path, collection_name=embedding_model_str)
            if embedding_model_str in embedding_models:
                embedding_model = embedding_models[embedding_model_str]
            else:
                logger.error(f"embedding_model_str {embedding_model_str} does not exist.")
                raise KeyError(f"embedding_model_str {embedding_model_str} does not exist.")
            ids, scores = vectordb(queries=queries, collection=chroma_collection,
                                   embedding_model=embedding_model, *args, **kwargs)
        else:
            raise ValueError(f"invalid f{retrieval_module} for using query_expansion_io decorator.")

        # fetch data from corpus_data
        corpus_data = pd.read_parquet(os.path.join(data_dir, "corpus.parquet"))
        contents = fetch_contents(corpus_data, ids)

        return decomposed_queries, contents, ids, scores

    return wrapper
