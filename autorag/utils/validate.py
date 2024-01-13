import pandas as pd


def validate_qa_dataset(df: pd.DataFrame):
    columns = ['qid', 'query', 'retrieval_gt', 'generation_gt']
    assert set(columns).issubset(df.columns), f"df must have columns {columns}, but got {df.columns}"


def validate_corpus_dataset(df: pd.DataFrame):
    columns = ['doc_id', 'contents', 'datetime', 'metadata']
    assert set(columns).issubset(df.columns), f"df must have columns {columns}, but got {df.columns}"
