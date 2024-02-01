import functools
from pathlib import Path
from typing import List, Union

import pandas as pd

from autorag.utils import result_to_dataframe


def prompt_maker_node(func):
    @functools.wraps(func)
    @result_to_dataframe(["prompts"])
    def wrapper(
            project_dir: Union[str, Path],
            previous_result: pd.DataFrame,
            *args, **kwargs) -> List[str]:
        # get query and retrieved contents from previous_result
        assert "query" in previous_result.columns, "previous_result must have query column."
        assert "retrieved_contents" in previous_result.columns, "previous_result must have retrieved_contents column."
        query = previous_result["query"].tolist()
        retrieved_contents = previous_result["retrieved_contents"].tolist()
        prompt = kwargs.pop("prompt")

        if func.__name__ == 'fstring':
            return func(prompt, query, retrieved_contents)
        else:
            raise NotImplementedError(f"Module {func.__name__} is not implemented or not supported.")

    return wrapper
