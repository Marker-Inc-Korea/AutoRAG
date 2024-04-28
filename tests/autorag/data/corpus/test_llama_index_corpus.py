import tempfile

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

from autorag.data.corpus import llama_documents_to_parquet, llama_text_node_to_parquet
from tests.autorag.data.corpus.test_base import validate_corpus


@pytest.fixture
def parquet_filepath():
    with tempfile.NamedTemporaryFile(suffix='.parquet') as temp_file:
        yield temp_file.name


def test_llama_documents_to_parquet(parquet_filepath):
    documents = [Document(text='test text', metadata={'key': 'value'},
                          ) for _ in range(5)]
    result_df = llama_documents_to_parquet(documents, parquet_filepath, upsert=True)
    validate_corpus(result_df, 5, parquet_filepath)


def test_llama_text_node_to_parquet(parquet_filepath):
    sample_text_node = TextNode(text='test text',
                                metadata={'key': 'value'},
                                relationships={NodeRelationship.PREVIOUS: RelatedNodeInfo(node_id='0'),
                                               NodeRelationship.NEXT: RelatedNodeInfo(node_id='2')})
    text_nodes = [sample_text_node for _ in range(5)]
    result_df = llama_text_node_to_parquet(text_nodes, parquet_filepath, upsert=True)
    validate_corpus(result_df, 5, parquet_filepath)
