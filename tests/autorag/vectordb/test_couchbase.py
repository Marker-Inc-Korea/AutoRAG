import asyncio
import os

import pytest

from autorag.vectordb.couchbase import Couchbase
from tests.delete_tests import is_github_action


@pytest.mark.skipif(
	is_github_action(),
	reason="This test needs couchbase connection string, username, and password",
)
@pytest.fixture
def couchbase_instance():
	couchbase = Couchbase(
		embedding_model="mock",
		bucket_name="autorag",
		scope_name="autorag",
		collection_name="autorag",
		index_name="autorag_search",
		connection_string=os.environ["COUCHBASE_CONNECTION_STRING"],
		username=os.environ["COUCHBASE_USERNAME"],
		password=os.environ["COUCHBASE_PASSWORD"],
	)
	yield couchbase


@pytest.mark.skipif(
	is_github_action(),
	reason="This test needs pinecone api key",
)
@pytest.mark.asyncio
async def test_add_and_query_documents(couchbase_instance):
	# Add documents
	ids = ["doc1", "doc2"]
	texts = ["This is a test document.", "This is another test document."]
	await couchbase_instance.add(ids, texts)

	await asyncio.sleep(1)

	# Query documents
	queries = ["test document"]
	contents, scores = await couchbase_instance.query(queries, top_k=2)

	assert len(contents) == 1
	assert len(scores) == 1
	assert len(contents[0]) == 2
	assert len(scores[0]) == 2
	assert scores[0][0] > scores[0][1]

	embeddings = await couchbase_instance.fetch([ids[0]])
	assert len(embeddings) == 1
	assert len(embeddings[0]) == 768

	exist = await couchbase_instance.is_exist([ids[0], "doc3"])
	assert len(exist) == 2
	assert exist[0] is True
	assert exist[1] is False


@pytest.mark.skipif(
	is_github_action(),
	reason="This test needs pinecone api key",
)
@pytest.mark.asyncio
async def test_delete_documents(couchbase_instance):
	# Add documents
	ids = ["doc1", "doc2"]
	texts = ["This is a test document.", "This is another test document."]
	await couchbase_instance.add(ids, texts)

	await asyncio.sleep(1)

	# Delete documents
	await couchbase_instance.delete([ids[0]])

	# Query documents to ensure they are deleted
	queries = ["test document"]
	contents, scores = await couchbase_instance.query(queries, top_k=2)

	assert len(contents[0]) == 1
	assert len(scores[0]) == 1
