import functools
import os
from pathlib import Path
from typing import List, Union, Tuple, Dict

import pandas as pd

from autorag import embedding_models, generator_models
from autorag.nodes.retrieval import bm25, vectordb
from autorag.nodes.retrieval.base import load_bm25_corpus, load_chroma_collection
from autorag.schema import Module
from autorag.utils import fetch_contents, result_to_dataframe, validate_qa_dataset

import logging

logger = logging.getLogger("AutoRAG")


def query_expansion_node(func):
    @functools.wraps(func)
    @result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
    def wrapper(
            project_dir: Union[str, Path],
            previous_result: pd.DataFrame,
            *args, **kwargs) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        validate_qa_dataset(previous_result)
        resources_dir = os.path.join(project_dir, "resources")
        data_dir = os.path.join(project_dir, "data")

        # find queries columns & type cast queries
        assert "query" in previous_result.columns, "previous_result must have query column."
        queries = previous_result["query"].tolist()

        # pop top_k from kwargs
        if "top_k" in kwargs.keys():
            top_k = kwargs.pop("top_k")
        else:
            top_k = 10  # default value

        # pop prompt from kwargs
        if "prompt" in kwargs.keys():
            prompt = kwargs.pop("prompt")
        else:
            prompt = ""

        # get retrieval module values
        if "retrieval_module" in kwargs.keys():
            retrieval_modules = kwargs.pop("retrieval_module")
        else:
            retrieval_modules = {"module_type": "bm25"}  # default value

        # set module parameters
        llm_str = kwargs.pop("llm")

        # set llm model for query expansion
        if llm_str in generator_models:
            llm = generator_models[llm_str](**kwargs)
        else:
            logger.error(f"llm_str {llm_str} does not exist.")
            raise KeyError(f"llm_str {llm_str} does not exist.")

        # run query expansion function
        if func.__name__ == "query_decompose":
            decomposed_queries = func(queries=queries, llm=llm, prompt=prompt)
        elif func.__name__ == "hyde":
            decomposed_queries = func(queries=queries, llm=llm, prompt=prompt)
            pass
        else:
            raise ValueError(f"Unknown query expansion function: {func.__name__}")

        # run retrieval function
        ids, scores = retrieval_by_retrieval_module(retrieval_module=retrieval_modules, resources_dir=resources_dir,
                                                    decomposed_queries=decomposed_queries, top_k=top_k)

        # fetch data from corpus_data
        corpus_data = pd.read_parquet(os.path.join(data_dir, "corpus.parquet"))
        contents = fetch_contents(corpus_data, ids)

        return contents, ids, scores

    return wrapper


def retrieval_by_retrieval_module(retrieval_module: Dict, resources_dir: str, decomposed_queries: List[List[str]],
                                  top_k: int) -> Tuple[List[List[str]], List[List[float]]]:
    module = Module.from_dict(retrieval_module)
    retrieval_module_type = module.module_type

    # set parameters for retrieval module
    if retrieval_module_type == "bm25":
        # check if bm25_path and file exists
        bm25_path = os.path.join(resources_dir, "resources", 'bm25.pkl')
        assert bm25_path is not None, "bm25_path must be specified for using bm25 retrieval."
        assert os.path.exists(bm25_path), f"bm25_path {bm25_path} does not exist. Please ingest first."

        # run retrieval function
        bm25_corpus = load_bm25_corpus(bm25_path)
        ids, scores = bm25(queries=decomposed_queries, bm25_corpus=bm25_corpus, top_k=top_k)

    elif retrieval_module_type == "vectordb":
        # check if chroma_path and file exists
        chroma_path = os.path.join(resources_dir, 'chroma')
        assert chroma_path is not None, "chroma_path must be specified for using vectordb retrieval."
        assert os.path.exists(chroma_path), f"chroma_path {chroma_path} does not exist. Please ingest first."

        # set embedding model
        embedding_model_str = module.module_param["embedding_model"]
        if embedding_model_str in embedding_models:
            embedding_model = embedding_models[embedding_model_str]
        else:
            logger.error(f"embedding_model_str {embedding_model_str} does not exist.")
            raise KeyError(f"embedding_model_str {embedding_model_str} does not exist.")

        # run retrieval function
        chroma_collection = load_chroma_collection(db_path=chroma_path, collection_name=embedding_model_str)
        original_vectordb = vectordb.__wrapped__
        ids, scores = original_vectordb(queries=decomposed_queries, collection=chroma_collection,
                                        embedding_model=embedding_model, top_k=top_k)
    else:
        raise ValueError(f"invalid f{retrieval_module} for using query_expansion_io decorator.")

    return ids, scores
