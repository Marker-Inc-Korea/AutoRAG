import pytest
from autorag.vectordb.chroma import Chroma


@pytest.fixture
def chroma_ephemeral():
	return Chroma(
		embedding_model="mock",
		collection_name="test_collection",
		client_type="ephemeral",
	)


@pytest.mark.asyncio
async def test_add_and_query_documents(chroma_ephemeral):
	# Add documents
	ids = ["doc1", "doc2"]
	texts = ["This is a test document.", "This is another test document."]
	await chroma_ephemeral.add(ids, texts)

	# Query documents
	queries = ["test document"]
	contents, scores = await chroma_ephemeral.query(queries, top_k=2)

	assert len(contents) == 1
	assert len(scores) == 1
	assert len(contents[0]) == 2
	assert len(scores[0]) == 2
	assert scores[0][0] > scores[0][1]

	embeddings = await chroma_ephemeral.fetch([ids[0]])
	assert len(embeddings) == 1
	assert len(embeddings[0]) == 768

	exist = await chroma_ephemeral.is_exist([ids[0], "doc3"])
	assert len(exist) == 2
	assert exist[0] is True
	assert exist[1] is False


@pytest.mark.asyncio
async def test_delete_documents(chroma_ephemeral):
	# Add documents
	ids = ["doc1", "doc2"]
	texts = ["This is a test document.", "This is another test document."]
	await chroma_ephemeral.add(ids, texts)

	# Delete documents
	await chroma_ephemeral.delete([ids[0]])

	# Query documents to ensure they are deleted
	queries = ["test document"]
	contents, scores = await chroma_ephemeral.query(queries, top_k=2)

	assert len(contents[0]) == 1
	assert len(scores[0]) == 1
