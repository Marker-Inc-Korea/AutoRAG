import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from qdrant_client.models import QueryRequest

from autorag.vectordb.qdrant import Qdrant
from tests.delete_tests import is_github_action


@pytest.mark.asyncio
async def test_query_uses_qdrant_query_points_batch_api():
	instance = Qdrant.__new__(Qdrant)
	instance.collection_name = "autorag_t"
	instance.embedding = SimpleNamespace(
		aget_text_embedding_batch=AsyncMock(return_value=[[0.1, 0.2]])
	)
	instance.truncated_inputs = lambda queries: queries
	instance._using_in_memory_store = lambda: False
	instance.client = Mock()
	instance.client.query_batch_points.return_value = [
		SimpleNamespace(points=[SimpleNamespace(id="doc-1", score=0.75)])
	]

	ids, scores = await instance.query(["test document"], top_k=2)

	request = instance.client.query_batch_points.call_args.kwargs["requests"][0]
	assert isinstance(request, QueryRequest)
	assert request.query == [0.1, 0.2]
	assert request.limit == 2
	assert request.with_vector is True
	assert ids == [["doc-1"]]
	assert scores == [[0.75]]


@pytest.fixture
def qdrant_instance():
	qdrant = Qdrant(
		embedding_model="mock",
		collection_name="autorag_t",
		client_type="docker",
		dimension=768,
	)
	yield qdrant
	qdrant.delete_collection()


@pytest.mark.asyncio
@pytest.mark.skipif(
	is_github_action(),
	reason="This test needs qdrant docker server.",
)
async def test_add_and_query_documents(qdrant_instance):
	# Add documents
	ids = [str(uuid.uuid4()) for _ in range(2)]
	texts = ["This is a test document.", "This is another test document."]
	await qdrant_instance.add(ids, texts)

	await asyncio.sleep(1)

	# Query documents
	queries = ["test document"]
	contents, scores = await qdrant_instance.query(queries, top_k=2)

	assert len(contents) == 1
	assert len(scores) == 1
	assert len(contents[0]) == 2
	assert len(scores[0]) == 2
	assert scores[0][0] > scores[0][1]

	embeddings = await qdrant_instance.fetch([ids[0]])
	assert len(embeddings) == 1
	assert len(embeddings[0]) == 768

	exist = await qdrant_instance.is_exist([ids[0], str(uuid.uuid4())])
	assert len(exist) == 2
	assert exist[0] is True
	assert exist[1] is False


@pytest.mark.asyncio
@pytest.mark.skipif(
	is_github_action(),
	reason="This test needs qdrant docker server.",
)
async def test_delete_documents(qdrant_instance):
	# Add documents
	ids = [str(uuid.uuid4()) for _ in range(2)]
	texts = ["This is a test document.", "This is another test document."]
	await qdrant_instance.add(ids, texts)

	await asyncio.sleep(1)

	# Delete documents
	await qdrant_instance.delete([ids[0]])

	# Query documents to ensure they are deleted
	queries = ["test document"]
	contents, scores = await qdrant_instance.query(queries, top_k=2)

	assert len(contents[0]) == 1
	assert len(scores[0]) == 1
