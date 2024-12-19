import logging

from random import random
from typing import List

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingModelType
from langchain_openai.embeddings import OpenAIEmbeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger("AutoRAG")

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

class EmbeddingModel:

    @staticmethod
    def load(name: str = "", **kwargs):
        
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
            "huggingface_bge_m3": LazyInit(
                HuggingFaceEmbedding, model_name="BAAI/bge-m3"
            ),
            # "huggingface_multilingual_e5_large": LazyInit(
            #     HuggingFaceEmbedding, model_name="intfloat/multilingual-e5-large-instruct"
            # ),
        }
        
        return embedding_models[name]()
    
    @staticmethod
    def load_from_dict(option: List[dict]):
        
        def _check_one_item(target: List):
            if len(target) > 1:
                raise ValueError("only one embedding model is supported")
            if len(target) == 0:
                raise ValueError("embedding model is empty")
            
        def _check_keys(target: dict):
            if target.get("type", None) not in ["openai", "huggingface", "mock"]:
                raise ValueError("type is not provided")
            if target.get("model_name", None) is None:
                raise ValueError("model_name is not provided")
       
        _check_one_item(option)
        _check_keys(option[0])

        _option = option[0]
        _type = _option.pop("type")
        # _model_name = _option.pop("model_name")
        
        _embedding = None
        
        if _type == "openai":
            _embedding = LazyInit(OpenAIEmbedding, **_option)
        elif _type == "huggingface":
            _embedding = LazyInit(HuggingFaceEmbedding, **_option)
        elif _type == "mock":
            _embedding = LazyInit(MockEmbeddingRandom, embed_dim=768)

        if _embedding is None :
            raise ValueError("type is not supported")            
        
        return _embedding
    