import tempfile

import pytest
from langchain_core.documents import Document

from autorag.data.corpus import langchain_documents_to_parquet
from tests.autorag.data.corpus.test_base import validate_corpus


@pytest.fixture
def parquet_filepath():
    with tempfile.NamedTemporaryFile(suffix='.parquet') as temp_file:
        yield temp_file.name


def test_langchain_documents_to_parquet(parquet_filepath):
    documents = [Document(page_content='test text', metadata={'key': 'value'}) for _ in range(5)]
    result_df = langchain_documents_to_parquet(documents, parquet_filepath, upsert=True)
    validate_corpus(result_df, 5, parquet_filepath)
