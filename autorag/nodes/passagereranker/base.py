import functools
import logging
import os
from pathlib import Path
from typing import List, Union, Tuple

import pandas as pd

from autorag.utils import result_to_dataframe, validate_qa_dataset, fetch_contents

logger = logging.getLogger("AutoRAG")


def passage_reranker_node(func):
    @functools.wraps(func)
    @result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
    def wrapper(
            project_dir: Union[str, Path],
            previous_result: pd.DataFrame,
            *args, **kwargs) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        logger.info(f"Running passage reranker node - {func.__name__} module...")
        validate_qa_dataset(previous_result)

        # find queries columns
        assert "query" in previous_result.columns, "previous_result must have query column."
        queries = previous_result["query"].tolist()

        # find contents_list columns
        assert "retrieved_contents" in previous_result.columns, "previous_result must have retrieved_contents column."
        contents = previous_result["retrieved_contents"].tolist()

        # find scores columns
        assert "retrieve_scores" in previous_result.columns, "previous_result must have retrieve_scores column."
        scores = previous_result["retrieve_scores"].tolist()

        # find ids columns
        assert "retrieved_ids" in previous_result.columns, "previous_result must have retrieved_ids column."
        ids = previous_result["retrieved_ids"].tolist()

        # time rerankers
        if func.__name__ == 'time_reranker':
            corpus_df = pd.read_parquet(os.path.join(project_dir, "data", "corpus.parquet"))
            metadatas = fetch_contents(corpus_df, ids, column_name='metadata')
            times = [[time['last_modified_datetime'] for time in time_list] for time_list in metadatas]
            reranked_contents, reranked_ids, reranked_scores \
                = func(contents_list=contents, scores_list=scores, ids_list=ids, time_list=times, *args, **kwargs)
        else:
            reranked_contents, reranked_ids, reranked_scores \
                = func(queries=queries, contents_list=contents, scores_list=scores, ids_list=ids, *args, **kwargs)

        return reranked_contents, reranked_ids, reranked_scores

    return wrapper
