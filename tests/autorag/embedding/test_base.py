import pytest
from llama_index.embeddings.openai import OpenAIEmbedding
from autorag.embedding.base import EmbeddingModel, MockEmbeddingRandom


def test_load_embedding_model():
	embedding = EmbeddingModel.load("mock")
	assert embedding is not None
	assert isinstance(embedding(), MockEmbeddingRandom)

	embedding = EmbeddingModel.load(
		[{"type": "openai", "model_name": "text-embedding-ada-002"}]
	)
	assert embedding is not None
	assert isinstance(embedding(), OpenAIEmbedding)


def test_load_from_str_embedding_model():
	# Test loading a supported embedding model
	embedding = EmbeddingModel.load_from_str("mock")
	assert embedding is not None
	assert isinstance(embedding(), MockEmbeddingRandom)

	# Test loading an unsupported embedding model
	with pytest.raises(
		ValueError, match="Embedding model 'unsupported_model' is not supported"
	):
		EmbeddingModel.load_from_str("unsupported_model")


def test_load_embedding_model_from_list():
	# Test loading with missing keys
	with pytest.raises(
		ValueError, match="Both 'type' and 'model_name' must be provided"
	):
		EmbeddingModel.load_from_list([{"type": "openai"}])

	# Test loading with an unsupported type
	with pytest.raises(
		ValueError, match="Embedding model type 'unsupported_type' is not supported"
	):
		EmbeddingModel.load_from_list(
			[{"type": "unsupported_type", "model_name": "some-model"}]
		)

	# Test loading with multiple items
	with pytest.raises(ValueError, match="Only one embedding model is supported"):
		EmbeddingModel.load_from_list(
			[
				{"type": "openai", "model_name": "text-embedding-ada-002"},
				{"type": "huggingface", "model_name": "BAAI/bge-small-en-v1.5"},
			]
		)

def test_load_embedding_model_from_dict():
	with pytest.raises(
		ValueError, match="Both 'type' and 'model_name' must be provided"
	):
		embedding = EmbeddingModel.load_from_dict({"type": "openai"})

	# Test loading with an unsupported type
	with pytest.raises(
		ValueError, match="Embedding model type 'unsupported_type' is not supported"
	):
		EmbeddingModel.load_from_dict(
			{"type": "unsupported_type", "model_name": "some-model"}
		)

	embedding = EmbeddingModel.load_from_dict({"type": "openai", "model_name": "text-embedding-ada-002"})
	assert embedding is not None
	assert isinstance(embedding(), OpenAIEmbedding)
