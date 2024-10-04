import os
import tempfile

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

from autorag.data.legacy.corpus import (
	llama_documents_to_parquet,
	llama_text_node_to_parquet,
)
from tests.autorag.data.legacy.corpus.test_base_corpus_legacy import validate_corpus
from tests.delete_tests import is_github_action


@pytest.fixture
def parquet_filepath():
	with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp_file:
		yield temp_file.name
		temp_file.close()
		os.unlink(temp_file.name)


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
def test_llama_documents_to_parquet(parquet_filepath):
	documents = [
		Document(
			text="test text",
			metadata={"key": "value"},
		)
		for _ in range(5)
	]
	result_df = llama_documents_to_parquet(documents, parquet_filepath, upsert=True)
	validate_corpus(result_df, 5, parquet_filepath)


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
def test_llama_text_node_to_parquet(parquet_filepath):
	sample_text_node = TextNode(
		text="test text",
		metadata={"key": "value"},
		relationships={
			NodeRelationship.PREVIOUS: RelatedNodeInfo(node_id="0"),
			NodeRelationship.NEXT: RelatedNodeInfo(node_id="2"),
		},
	)
	text_nodes = [sample_text_node for _ in range(5)]
	result_df = llama_text_node_to_parquet(text_nodes, parquet_filepath, upsert=True)
	validate_corpus(result_df, 5, parquet_filepath)
