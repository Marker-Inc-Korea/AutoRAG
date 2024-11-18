import asyncio
import os

import pytest

from autorag.vectordb.pinecone import Pinecone
from tests.delete_tests import is_github_action


@pytest.mark.skipif(
	is_github_action(),
	reason="This test needs pinecone api key",
)
@pytest.fixture
def pinecone_instance():
	pinecone = Pinecone(
		embedding_model="mock",
		index_name="nodonggeon",
		dimension=768,  # mock embedding model has 768 dimensions
		api_key=os.environ["PINECONE_API_KEY"],
	)
	yield pinecone
	pinecone.delete_index()


@pytest.mark.skipif(
	is_github_action(),
	reason="This test needs pinecone api key",
)
@pytest.mark.asyncio
async def test_add_and_query_documents(pinecone_instance):
	# Add documents
	ids = ["doc1", "doc2"]
	texts = ["This is a test document.", "This is another test document."]
	await pinecone_instance.add(ids, texts)

	await asyncio.sleep(10)

	# Query documents
	queries = ["test document"]
	contents, scores = await pinecone_instance.query(queries, top_k=2)

	assert len(contents) == 1
	assert len(scores) == 1
	assert len(contents[0]) == 2
	assert len(scores[0]) == 2
	assert scores[0][0] > scores[0][1]

	embeddings = await pinecone_instance.fetch([ids[0]])
	assert len(embeddings) == 1
	assert len(embeddings[0]) == 768

	exist = await pinecone_instance.is_exist([ids[0], "doc3"])
	assert len(exist) == 2
	assert exist[0] is True
	assert exist[1] is False


@pytest.mark.skipif(
	is_github_action(),
	reason="This test needs pinecone api key",
)
@pytest.mark.asyncio
async def test_delete_documents(pinecone_instance):
	# Add documents
	ids = ["doc1", "doc2"]
	texts = ["This is a test document.", "This is another test document."]
	await pinecone_instance.add(ids, texts)

	await asyncio.sleep(10)

	# Delete documents
	await pinecone_instance.delete([ids[0]])

	# Query documents to ensure they are deleted
	queries = ["test document"]
	contents, scores = await pinecone_instance.query(queries, top_k=2)

	assert len(contents[0]) == 1
	assert len(scores[0]) == 1
