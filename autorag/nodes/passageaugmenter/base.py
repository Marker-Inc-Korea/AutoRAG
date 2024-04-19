import functools
import itertools
import logging
import os
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import torch

from autorag import embedding_models
from autorag.evaluate.metric.util import calculate_cosine_similarity
from autorag.utils import result_to_dataframe, validate_qa_dataset, fetch_contents, sort_by_scores
from autorag.utils.util import reconstruct_list, filter_dict_keys

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

        corpus_df = pd.read_parquet(os.path.join(data_dir, "corpus.parquet"))

        if func.__name__ == 'prev_next_augmenter':
            slim_corpus_df = corpus_df[["doc_id", "metadata"]]
            slim_corpus_df['metadata'] = slim_corpus_df['metadata'].apply(filter_dict_keys, keys=['prev_id', 'next_id'])

            mode = kwargs.pop("mode", 'next')
            num_passages = kwargs.pop("num_passages", 1)

            # get augmented ids
            ids = func(ids_list=ids, corpus_df=slim_corpus_df, mode=mode, num_passages=num_passages)
        else:
            ids = func(ids_list=ids, *args, **kwargs)

        # fetch contents from corpus to use augmented ids
        contents = fetch_contents(corpus_df, ids)

        # set embedding model for getting scores
        embedding_model_str = kwargs.pop("embedding_model", 'openai')
        query_embeddings, contents_embeddings = embedding_query_content(queries, contents, embedding_model_str,
                                                                        batch=128)

        # get scores from calculated cosine similarity
        scores = [np.array([calculate_cosine_similarity(query_embedding, x) for x in content_embeddings]).tolist()
                  for query_embedding, content_embeddings in zip(query_embeddings, contents_embeddings)]

        # sort by scores
        df = pd.DataFrame({
            'contents': contents,
            'ids': ids,
            'scores': scores,
        })
        df[['contents', 'ids', 'scores']] = df.apply(sort_by_scores, axis=1, result_type='expand')
        augmented_contents, augmented_ids, augmented_scores = \
            df['contents'].tolist(), df['ids'].tolist(), df['scores'].tolist()

        return augmented_contents, augmented_ids, augmented_scores

    return wrapper


def embedding_query_content(queries: List[str], contents_list: List[List[str]],
                            embedding_model: str, batch: int = 128):
    embedding_model = embedding_models[embedding_model]

    # Embedding using batch
    embedding_model.embed_batch_size = batch
    query_embeddings = embedding_model.get_text_embedding_batch(queries)

    content_lengths = list(map(len, contents_list))
    content_embeddings_flatten = embedding_model.get_text_embedding_batch(list(
        itertools.chain.from_iterable(contents_list)))
    content_embeddings = reconstruct_list(content_embeddings_flatten, content_lengths)

    del embedding_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return query_embeddings, content_embeddings
