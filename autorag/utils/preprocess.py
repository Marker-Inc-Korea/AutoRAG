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

    return df
