import logging
import sys

from random import random
from typing import List, Union, Dict

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_openai.embeddings import OpenAIEmbeddings
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

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
	"ollama": LazyInit(OllamaEmbedding),
	# openai like
	"openai_like": LazyInit(OpenAILikeEmbedding),
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
	def load(config: Union[str, Dict, List[Dict]]):
		if isinstance(config, str):
			return EmbeddingModel.load_from_str(config)
		elif isinstance(config, dict):
			return EmbeddingModel.load_from_dict(config)
		elif isinstance(config, list):
			return EmbeddingModel.load_from_list(config)
		else:
			raise ValueError("Invalid type of config")

	@staticmethod
	def load_from_str(name: str):
		try:
			return embedding_models[name]
		except KeyError:
			raise ValueError(f"Embedding model '{name}' is not supported")

	@staticmethod
	def load_from_list(option: List[dict]):
		if len(option) != 1:
			raise ValueError("Only one embedding model is supported")
		return EmbeddingModel.load_from_dict(option[0])

	@staticmethod
	def load_from_dict(option: dict):
		def _check_keys(target: dict):
			if "type" not in target or "model_name" not in target:
				raise ValueError("Both 'type' and 'model_name' must be provided")
			if target["type"] not in ["openai", "huggingface", "mock", "ollama"]:
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

		_check_keys(option)

		model_options = option
		model_type = model_options.pop("type")

		embedding_map = {
			"openai": OpenAIEmbedding,
			"mock": MockEmbeddingRandom,
			"huggingface": _get_huggingface_class(),
			"ollama": OllamaEmbedding,
			"openai_like": OpenAILikeEmbedding,
		}

		embedding_class = embedding_map.get(model_type)
		if not embedding_class:
			raise ValueError(f"Embedding model type '{model_type}' is not supported")

		return LazyInit(embedding_class, **model_options)
