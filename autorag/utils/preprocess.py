from datetime import datetime

import numpy as np
import pandas as pd


def validate_qa_dataset(df: pd.DataFrame):
    columns = ['qid', 'query', 'retrieval_gt', 'generation_gt']
    assert set(columns).issubset(df.columns), f"df must have columns {columns}, but got {df.columns}"


def validate_corpus_dataset(df: pd.DataFrame):
    columns = ['doc_id', 'contents', 'metadata']
    assert set(columns).issubset(df.columns), f"df must have columns {columns}, but got {df.columns}"


def cast_qa_dataset(df: pd.DataFrame):
    def cast_retrieval_gt(gt):
        if isinstance(gt, str):
            return [[gt]]
        elif isinstance(gt, list):
            if isinstance(gt[0], str):
                return [gt]
            elif isinstance(gt[0], list):
                return gt
            elif isinstance(gt[0], np.ndarray):
                return cast_retrieval_gt(list(map(lambda x: x.tolist(), gt)))
            else:
                raise ValueError(f"retrieval_gt must be str or list, but got {type(gt[0])}")
        elif isinstance(gt, np.ndarray):
            return cast_retrieval_gt(gt.tolist())
        else:
            raise ValueError(f"retrieval_gt must be str or list, but got {type(gt)}")

    def cast_generation_gt(gt):
        if isinstance(gt, str):
            return [gt]
        elif isinstance(gt, list):
            return gt
        elif isinstance(gt, np.ndarray):
            return cast_generation_gt(gt.tolist())
        else:
            raise ValueError(f"generation_gt must be str or list, but got {type(gt)}")

    validate_qa_dataset(df)
    assert df['qid'].apply(lambda x: isinstance(x, str)).sum() == len(df), \
        "qid must be string type."
    assert df['query'].apply(lambda x: isinstance(x, str)).sum() == len(df), \
        "query must be string type."
    df['retrieval_gt'] = df['retrieval_gt'].apply(cast_retrieval_gt)
    df['generation_gt'] = df['generation_gt'].apply(cast_generation_gt)
    return df


def cast_corpus_dataset(df: pd.DataFrame):
    validate_corpus_dataset(df)

    def make_datetime_metadata(x):
        if x is None or x == {}:
            return {'last_modified_datetime': datetime.now()}
        elif x.get('last_modified_datetime') is None:
            return {**x, 'last_modified_datetime': datetime.now()}
        else:
            return x

    df['metadata'] = df['metadata'].apply(make_datetime_metadata)

    # check every metadata have a datetime key
    assert sum(df['metadata'].apply(lambda x: x.get('last_modified_datetime') is not None)) == len(df), \
        "Every metadata must have a datetime key."

    def make_prev_next_id_metadata(row, prev_doc_id, next_doc_id):
        metadata = row['metadata']
        if metadata is None or metadata == {}:
            return {'prev_id': prev_doc_id, 'next_id': next_doc_id}
        metadata.setdefault('prev_id', prev_doc_id)
        metadata.setdefault('next_id', next_doc_id)
        return metadata

    temp_prev_df = pd.DataFrame({'prev_doc_id': df['doc_id'].shift(1)})
    temp_next_df = pd.DataFrame({'next_doc_id': df['doc_id'].shift(-1)})
    df['metadata'] = df.apply(lambda row: make_prev_next_id_metadata(row, temp_prev_df.loc[row.name, 'prev_doc_id'],
                                                                     temp_next_df.loc[row.name, 'next_doc_id']), axis=1)
    # check every metadata have a prev_id, next_id key
    assert all('prev_id' in metadata for metadata in df['metadata']), "Every metadata must have a prev_id key."
    assert all('next_id' in metadata for metadata in df['metadata']), "Every metadata must have a next_id key."

    return df
