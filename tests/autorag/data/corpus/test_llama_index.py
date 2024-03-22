import os
import tempfile

import pandas as pd
import pytest
from llama_index.core import Document

from autorag.data.corpus import llama_documents_to_parquet


@pytest.fixture
def parquet_filepath():
    with tempfile.NamedTemporaryFile(suffix='.parquet') as temp_file:
        yield temp_file.name


def test_llama_documents_to_parquet(parquet_filepath):
    documents = [Document(text='test text', metadata={'key': 'value'}) for _ in range(5)]

    result_df = llama_documents_to_parquet(documents, parquet_filepath, upsert=True)

    assert isinstance(result_df, pd.DataFrame)
    assert 'doc_id' in result_df.columns
    assert 'contents' in result_df.columns
    assert 'metadata' in result_df.columns
    assert os.path.exists(parquet_filepath)

    assert ['test text'] * 5 == result_df['contents'].tolist()
    assert all(['last_modified_datetime' in metadata for metadata in result_df['metadata'].tolist()])
    assert all([isinstance(doc_id, str) for doc_id in result_df['doc_id'].tolist()])
