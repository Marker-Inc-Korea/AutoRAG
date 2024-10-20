import pytest
from autorag.vectordb.chroma import Chroma


@pytest.fixture
def chroma_ephemeral():
	return Chroma(
		embedding_model="mock",
		collection_name="test_collection",
		client_type="ephemeral",
	)


def test_add_and_query_documents(chroma_ephemeral):
	# Add documents
	ids = ["doc1", "doc2"]
	texts = ["This is a test document.", "This is another test document."]
	chroma_ephemeral.add(ids, texts)

	# Query documents
	queries = ["test document"]
	contents, scores = chroma_ephemeral.query(queries, top_k=2)

	assert len(contents) == 1
	assert len(scores) == 1
	assert len(contents[0]) == 2
	assert len(scores[0]) == 2
	assert scores[0][0] > scores[0][1]

	contents = chroma_ephemeral.fetch([ids[0]])
	assert len(contents) == 1
	assert contents[0] == texts[0]


def test_delete_documents(chroma_ephemeral):
	# Add documents
	ids = ["doc1", "doc2"]
	texts = ["This is a test document.", "This is another test document."]
	chroma_ephemeral.add(ids, texts)

	# Delete documents
	chroma_ephemeral.delete([ids[0]])

	# Query documents to ensure they are deleted
	queries = ["test document"]
	contents, scores = chroma_ephemeral.query(queries, top_k=2)

	assert len(contents[0]) == 1
	assert len(scores[0]) == 1
