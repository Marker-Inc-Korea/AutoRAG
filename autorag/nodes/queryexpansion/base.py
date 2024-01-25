import functools
from pathlib import Path
from typing import List, Union

import pandas as pd

from autorag.utils import result_to_dataframe, validate_qa_dataset

import logging
logger = logging.getLogger("AutoRAG")


def query_expansion_node(func):

    @functools.wraps(func)
    @result_to_dataframe(["expanded_queries"])
    def wrapper(
            project_dir: Union[str, Path],
            previous_result: pd.DataFrame,
            *args, **kwargs) -> List[List[str]]:
        validate_qa_dataset(previous_result)

        # find queries columns & type cast queries
        assert "query" in previous_result.columns, "previous_result must have query column."
        queries = previous_result["query"].tolist()

        # run query expansion function
        if func.__name__ == "query_decompose":
            decomposed_queries = func(queries=queries, *args, **kwargs)
        elif func.__name__ == "hyde":
            # TODO: implement hyde
            # decomposed_queries = func(queries=queries, *args, **kwargs)
            pass
        else:
            raise ValueError(f"Unknown query expansion function: {func.__name__}")

        return decomposed_queries

    return wrapper

