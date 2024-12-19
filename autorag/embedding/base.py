from random import random
from typing import List

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
    def load_from_dict(**kwargs):
        if kwargs.get("type", None) == None:
            raise ValueError("type is not provided")
        if kwargs.get("model_name", None) == None:
            raise ValueError("model_name is not provided")
        
        _type = kwargs.get("type")
        _embedding = None
        
        if _type == "openai":
            _embedding = LazyInit(OpenAIEmbedding, kwargs=kwargs)
        elif _type == "huggingface":
            _embedding = LazyInit(HuggingFaceEmbedding, kwargs=kwargs)
        elif _type == "mock":
            _embedding = LazyInit(MockEmbeddingRandom, embed_dim=768)

        if _embedding is None :
            raise ValueError("type is not supported")            
        
        return _embedding