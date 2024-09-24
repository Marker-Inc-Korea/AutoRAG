import logging
import logging.config
import os
import sys
from random import random
from typing import List

import transformers
from llama_index.core import MockEmbedding
from llama_index.core.llms.mock import MockLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from langchain_openai.embeddings import OpenAIEmbeddings
from rich.logging import RichHandler

version_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "VERSION")

with open(version_path, "r") as f:
	__version__ = f.read().strip()


class LazyInit:
	def __init__(self, factory, *args, **kwargs):
		self._factory = factory
		self._args = args
		self._kwargs = kwargs
		self._instance = None

	def __call__(self):
		if self._instance is None:
			self._instance = self._factory(*self._args, **self._kwargs)
		return self._instance

	def __getattr__(self, name):
		if self._instance is None:
			self._instance = self._factory(*self._args, **self._kwargs)
		return getattr(self._instance, name)


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
	# you can use your own model in this way.
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
	"mock": LazyInit(MockEmbeddingRandom, embed_dim=768),
	# langchain
	"openai_langchain": LazyInit(OpenAIEmbeddings),
}

generator_models = {
	"openai": OpenAI,
	"huggingfacellm": HuggingFaceLLM,
	"openailike": OpenAILike,
	"ollama": Ollama,
	"mock": MockLLM,
}

rich_format = "[%(filename)s:%(lineno)s] >> %(message)s"
logging.basicConfig(
	level="INFO", format=rich_format, handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("AutoRAG")


def handle_exception(exc_type, exc_value, exc_traceback):
	logger = logging.getLogger("AutoRAG")
	logger.error("Unexpected exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

transformers.logging.set_verbosity_error()
