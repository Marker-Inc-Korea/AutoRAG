import logging
import uuid
from typing import Callable, Optional

import pandas as pd
from tqdm import tqdm

from autorag.utils.util import save_parquet_safe

logger = logging.getLogger("AutoRAG")


def make_single_content_qa(corpus_df: pd.DataFrame,
                           content_size: int,
                           qa_creation_func: Callable,
                           output_filepath: Optional[str] = None,
                           upsert: bool = False,
                           random_state: int = 42,
                           cache_batch: int = 32,
                           **kwargs) -> pd.DataFrame:
    """
    Make single content (single-hop, single-document) QA dataset using given qa_creation_func.
    It generates a single content QA dataset, which means its retrieval ground truth will be only one.
    It is the most basic form of QA dataset.

    :param corpus_df: The corpus dataframe to make QA dataset from.
    :param content_size: This function will generate QA dataset for the given number of contents.
    :param qa_creation_func: The function to create QA pairs.
        You can use like `generate_qa_llama_index` or `generate_qa_llama_index_by_ratio`.
        The input func must have `contents` parameter for the list of content string.
    :param output_filepath: Optional filepath to save the parquet file.
        If None, the function will return the processed_data as pd.DataFrame, but do not save as parquet.
        File directory must exist. File extension must be .parquet
    :param upsert: If true, the function will overwrite the existing file if it exists.
        Default is False.
    :param random_state: The random state for sampling corpus from the given corpus_df.
    :param cache_batch: The number of batches to use for caching the generated QA dataset.
        When the cache_batch size data is generated, the dataset will save to the designated output_filepath.
        If the cache_batch size is too small, the process time will be longer.
    :param kwargs: The keyword arguments for qa_creation_func.
    :return: QA dataset dataframe.
        You can save this as parquet file to use at AutoRAG.
    """
    assert content_size > 0, "content_size must be greater than 0."
    if content_size > len(corpus_df):
        logger.warning(f"content_size {content_size} is larger than the corpus size {len(corpus_df)}. "
                       "Setting content_size to the corpus size.")
        content_size = len(corpus_df)
    sampled_corpus = corpus_df.sample(n=content_size, random_state=random_state)
    sampled_corpus = sampled_corpus.reset_index(drop=True)

    def make_query_generation_gt(row):
        return row['qa']['query'], row['qa']['generation_gt']

    qa_data = pd.DataFrame()
    for idx, i in tqdm(enumerate(range(0, len(sampled_corpus), cache_batch))):
        qa = qa_creation_func(contents=sampled_corpus['contents'].tolist()[i:i + cache_batch], **kwargs)

        temp_qa_data = pd.DataFrame({
            'qa': qa,
            'retrieval_gt': sampled_corpus['doc_id'].tolist()[i:i + cache_batch],
        })
        temp_qa_data = temp_qa_data.explode('qa', ignore_index=True)
        temp_qa_data['qid'] = [str(uuid.uuid4()) for _ in range(len(temp_qa_data))]
        temp_qa_data[['query', 'generation_gt']] = temp_qa_data.apply(make_query_generation_gt, axis=1,
                                                                      result_type='expand')
        temp_qa_data = temp_qa_data.drop(columns=['qa'])

        temp_qa_data['retrieval_gt'] = temp_qa_data['retrieval_gt'].apply(lambda x: [[x]])
        temp_qa_data['generation_gt'] = temp_qa_data['generation_gt'].apply(lambda x: [x])

        if idx == 0:
            qa_data = temp_qa_data
        else:
            qa_data = pd.concat([qa_data, temp_qa_data], ignore_index=True)
        if output_filepath is not None:
            save_parquet_safe(qa_data, output_filepath, upsert=upsert)

    return qa_data
