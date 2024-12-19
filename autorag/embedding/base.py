import logging

from random import random
from typing import List

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from langchain_openai.embeddings import OpenAIEmbeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from autorag import LazyInit

logger = logging.getLogger("AutoRAG")

class MockEmbeddingRandom(MockEmbedding):
	"""Mock embedding with random vectors."""

	def _get_vector(self) -> List[float]:
		return [random() for _ in range(self.embed_dim)]


embedding_models = {
    # llama index
    "openai": LazyInit(
        OpenAIEmbedding
    ),  # default model is OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002
    "openai_embed_3_large": LazyInit(
        OpenAIEmbedding, model_name=OpenAIEmbeddingModelType.TEXT_EMBED_3_LARGE
    ),
    "openai_embed_3_small": LazyInit(
        OpenAIEmbedding, model_name=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL
    ),
    "mock": LazyInit(MockEmbeddingRandom, embed_dim=768),
    # langchain
    "openai_langchain": LazyInit(OpenAIEmbeddings),
    # local model
    "huggingface_baai_bge_small": LazyInit(
        HuggingFaceEmbedding, model_name="BAAI/bge-small-en-v1.5"
    ),
    "huggingface_cointegrated_rubert_tiny2": LazyInit(
        HuggingFaceEmbedding, model_name="cointegrated/rubert-tiny2"
    ),
    "huggingface_all_mpnet_base_v2": LazyInit(
        HuggingFaceEmbedding,
        model_name="sentence-transformers/all-mpnet-base-v2",
        max_length=512,
    ),
    "huggingface_bge_m3": LazyInit(HuggingFaceEmbedding, model_name="BAAI/bge-m3"),
    "huggingface_multilingual_e5_large": LazyInit(
        HuggingFaceEmbedding, model_name="intfloat/multilingual-e5-large-instruct"
    ),
}


class EmbeddingModel:

	@staticmethod
	def load(name: str = ""):
		try:
			return embedding_models[name]()
		except KeyError:
			raise ValueError(f"Embedding model '{name}' is not supported")

	@staticmethod
	def load_from_dict(option: List[dict]):
		def _check_one_item(target: List):
			if len(target) != 1:
				raise ValueError("Only one embedding model is supported")

		def _check_keys(target: dict):
			if "type" not in target or "model_name" not in target:
				raise ValueError("Both 'type' and 'model_name' must be provided")
			if target["type"] not in ["openai", "huggingface", "mock"]:
				raise ValueError(
					f"Embedding model type '{target['type']}' is not supported"
				)

		_check_one_item(option)
		_check_keys(option[0])

		model_options = option[0]
		model_type = model_options.pop("type")

		embedding_map = {
			"openai": OpenAIEmbedding,
			"huggingface": HuggingFaceEmbedding,
			"mock": MockEmbeddingRandom,
		}

		embedding_class = embedding_map.get(model_type)
		if not embedding_class:
			raise ValueError(f"Embedding model type '{model_type}' is not supported")

		return LazyInit(embedding_class, **model_options)
