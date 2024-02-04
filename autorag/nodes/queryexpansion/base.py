import functools
from pathlib import Path
from typing import List, Union

import pandas as pd

from autorag import generator_models

from autorag.utils import result_to_dataframe, validate_qa_dataset

import logging

logger = logging.getLogger("AutoRAG")


def query_expansion_node(func):
    @functools.wraps(func)
    @result_to_dataframe(["queries"])
    def wrapper(
            project_dir: Union[str, Path],
            previous_result: pd.DataFrame,
            *args, **kwargs) -> List[List[str]]:
        validate_qa_dataset(previous_result)

        # find queries columns
        assert "query" in previous_result.columns, "previous_result must have query column."
        queries = previous_result["query"].tolist()

        # set module parameters
        llm_str = kwargs.pop("llm")

        # pop prompt from kwargs
        if "prompt" in kwargs.keys():
            prompt = kwargs.pop("prompt")
        else:
            prompt = ""

        # pop batch from kwargs
        if "batch" in kwargs.keys():
            batch = kwargs.pop("batch")
        else:
            batch = 16

        # set llm model for query expansion
        if llm_str in generator_models:
            llm = generator_models[llm_str](**kwargs)
        else:
            logger.error(f"llm_str {llm_str} does not exist.")
            raise KeyError(f"llm_str {llm_str} does not exist.")

        # run query expansion function
        expanded_queries = func(queries=queries, llm=llm, prompt=prompt, batch=batch)

        return expanded_queries

    return wrapper



