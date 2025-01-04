import logging
import sys

from random import random
from typing import List

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from langchain_openai.embeddings import OpenAIEmbeddings

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
}

try:
	# you can use your own model in this way.
	from llama_index.embeddings.huggingface import HuggingFaceEmbedding

	embedding_models["huggingface_baai_bge_small"] = LazyInit(
		HuggingFaceEmbedding, model_name="BAAI/bge-small-en-v1.5"
	)
	embedding_models["huggingface_cointegrated_rubert_tiny2"] = LazyInit(
		HuggingFaceEmbedding, model_name="cointegrated/rubert-tiny2"
	)
	embedding_models["huggingface_all_mpnet_base_v2"] = LazyInit(
		HuggingFaceEmbedding,
		model_name="sentence-transformers/all-mpnet-base-v2",
		max_length=512,
	)
	embedding_models["huggingface_bge_m3"] = LazyInit(
		HuggingFaceEmbedding, model_name="BAAI/bge-m3"
	)
	embedding_models["huggingface_multilingual_e5_large"] = LazyInit(
		HuggingFaceEmbedding, model_name="intfloat/multilingual-e5-large-instruct"
	)
except ImportError:
	logger.info(
		"You are using API version of AutoRAG."
		"To use local version, run pip install 'AutoRAG[gpu]'"
	)


class EmbeddingModel:
	@staticmethod
	def load(name: str = ""):
		try:
			return embedding_models[name]()
		except KeyError:
			raise ValueError(f"Embedding model '{name}' is not supported")

	@staticmethod
	def load_from_dict(option: List[dict]):
		def _check_keys(target: dict):
			if "type" not in target or "model_name" not in target:
				raise ValueError("Both 'type' and 'model_name' must be provided")
			if target["type"] not in ["openai", "huggingface", "mock"]:
				raise ValueError(
					f"Embedding model type '{target['type']}' is not supported"
				)

		def _get_huggingface_class():
			module = sys.modules.get("llama_index.embeddings.huggingface")
			if not module:
				logger.info(
					"You are using API version of AutoRAG. "
					"To use local version, run `pip install 'AutoRAG[gpu]'`."
				)
				return None
			return getattr(module, "HuggingFaceEmbedding", None)

		assert len(option) != 1, "Only one embedding model is supported"
		_check_keys(option[0])

		model_options = option[0]
		model_type = model_options.pop("type")

		embedding_map = {
			"openai": OpenAIEmbedding,
			"mock": MockEmbeddingRandom,
			"huggingface": _get_huggingface_class(),
		}

		embedding_class = embedding_map.get(model_type)
		if not embedding_class:
			raise ValueError(f"Embedding model type '{model_type}' is not supported")

		return LazyInit(embedding_class, **model_options)
