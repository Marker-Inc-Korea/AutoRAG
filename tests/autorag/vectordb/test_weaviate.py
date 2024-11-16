import asyncio
import uuid

import pytest

from autorag.vectordb.weaviate import Weaviate
from tests.delete_tests import is_github_action


@pytest.mark.skipif(
	is_github_action(),
	reason="This test needs docker.",
)
@pytest.fixture
def weaviate_instance():
	weaviate = Weaviate(
		embedding_model="mock",
		collection_name="test_collection",
		similarity_metric="cosine",
	)
	yield weaviate
	weaviate.delete_collection()


@pytest.mark.skipif(
	is_github_action(),
	reason="This test needs docker.",
)
@pytest.mark.asyncio
async def test_add_and_query_documents(weaviate_instance):
	# Add documents
	ids = [str(uuid.uuid4()) for _ in range(2)]
	texts = ["This is a test document.", "This is another test document."]
	await weaviate_instance.add(ids, texts)

	await asyncio.sleep(1)

	# Query documents
	queries = ["test document"]
	contents, scores = await weaviate_instance.query(queries, top_k=2)

	assert len(contents) == 1
	assert len(scores) == 1
	assert len(contents[0]) == 2
	assert len(scores[0]) == 2
	assert scores[0][0] > scores[0][1]

	embeddings = await weaviate_instance.fetch([ids[0]])
	assert len(embeddings) == 1
	assert len(embeddings[0]) == 768

	exist = await weaviate_instance.is_exist([ids[0], "doc3"])
	assert len(exist) == 2
	assert exist[0] is True
	assert exist[1] is False


@pytest.mark.skipif(
	is_github_action(),
	reason="This test needs docker.",
)
@pytest.mark.asyncio
async def test_delete_documents(weaviate_instance):
	# Add documents
	ids = [str(uuid.uuid4()) for _ in range(2)]
	texts = ["This is a test document.", "This is another test document."]
	await weaviate_instance.add(ids, texts)

	await asyncio.sleep(1)

	# Delete documents
	await weaviate_instance.delete([ids[0]])

	# Query documents to ensure they are deleted
	queries = ["test document"]
	contents, scores = await weaviate_instance.query(queries, top_k=2)

	assert len(contents[0]) == 1
	assert len(scores[0]) == 1
