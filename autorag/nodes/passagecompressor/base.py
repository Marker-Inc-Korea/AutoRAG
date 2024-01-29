import functools
from pathlib import Path
from typing import List, Union

import pandas as pd

from autorag.utils import result_to_dataframe


def passage_compressor_node(func):
    @functools.wraps(func)
    @result_to_dataframe(['retrieved_contents'])
    def wrapper(
            project_dir: Union[str, Path],
            previous_result: pd.DataFrame,
            *args, **kwargs) -> List[List[str]]:
        assert all([column in previous_result.columns for column in
                    ['query', 'retrieved_contents', 'retrieved_ids', 'retrieve_scores']]), \
            "previous_result must have retrieved_contents, retrieved_ids, and retrieve_scores columns."
        assert len(previous_result) > 0, "previous_result must have at least one row."

        queries = previous_result['query'].tolist()
        retrieved_contents = previous_result['retrieved_contents'].tolist()
        retrieved_ids = previous_result['retrieved_ids'].tolist()
        retrieve_scores = previous_result['retrieve_scores'].tolist()

        return list(map(lambda x: [x], func(
            queries=queries,
            contents=retrieved_contents,
            ids=retrieved_ids,
            scores=retrieve_scores,
            *args, **kwargs
        )))

    return wrapper
