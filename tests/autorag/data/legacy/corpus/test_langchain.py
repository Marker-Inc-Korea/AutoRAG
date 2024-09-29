import os
import tempfile

import pytest
from langchain_core.documents import Document

from autorag.data.legacy.corpus import langchain_documents_to_parquet
from tests.autorag.data.legacy.corpus.test_base_corpus_legacy import validate_corpus


@pytest.fixture
def parquet_filepath():
	with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
		yield temp_file.name
		temp_file.close()
		os.unlink(temp_file.name)


def test_langchain_documents_to_parquet(parquet_filepath):
	documents = [
		Document(page_content="test text", metadata={"key": "value"}) for _ in range(5)
	]
	result_df = langchain_documents_to_parquet(documents, parquet_filepath, upsert=True)
	validate_corpus(result_df, 5, parquet_filepath)
