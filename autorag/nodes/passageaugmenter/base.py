import functools
import logging
import os
from pathlib import Path
from typing import List, Union, Tuple

import pandas as pd

from autorag.utils import result_to_dataframe, validate_qa_dataset, fetch_contents, get_cosine_similarity_scores

logger = logging.getLogger("AutoRAG")


def passage_augmenter_node(func):
    @functools.wraps(func)
    @result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
    def wrapper(
            project_dir: Union[str, Path],
            previous_result: pd.DataFrame,
            *args, **kwargs) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        validate_qa_dataset(previous_result)
        data_dir = os.path.join(project_dir, "data")

        # find queries columns
        assert "query" in previous_result.columns, "previous_result must have query column."
        queries = previous_result["query"].tolist()

        # find ids columns
        assert "retrieved_ids" in previous_result.columns, "previous_result must have retrieved_ids column."
        ids = previous_result["retrieved_ids"].tolist()

        corpus_data = pd.read_parquet(os.path.join(data_dir, "corpus.parquet"))
        all_ids_list = corpus_data['doc_id'].tolist()

        augmented_ids = func(ids_list=ids, all_ids_list=all_ids_list, *args, **kwargs)

        augmented_contents = fetch_contents(corpus_data, augmented_ids)

        augmented_scores = get_cosine_similarity_scores(queries, augmented_contents)

        return augmented_contents, augmented_ids, augmented_scores

    return wrapper
