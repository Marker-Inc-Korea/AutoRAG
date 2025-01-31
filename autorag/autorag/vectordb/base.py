from abc import abstractmethod
from typing import List, Tuple, Union

from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.utils.util import openai_truncate_by_token
from autorag.embedding.base import EmbeddingModel


class BaseVectorStore:
	support_similarity_metrics = ["l2", "ip", "cosine"]

	def __init__(
		self,
		embedding_model: Union[str, List[dict]],
		similarity_metric: str = "cosine",
		embedding_batch: int = 100,
	):
		self.embedding = EmbeddingModel.load(embedding_model)()
		self.embedding_batch = embedding_batch
		self.embedding.embed_batch_size = embedding_batch
		assert (
			similarity_metric in self.support_similarity_metrics
		), f"search method {similarity_metric} is not supported"
		self.similarity_metric = similarity_metric

	@abstractmethod
	async def add(
		self,
		ids: List[str],
		texts: List[str],
	):
		pass

	@abstractmethod
	async def query(
		self, queries: List[str], top_k: int, **kwargs
	) -> Tuple[List[List[str]], List[List[float]]]:
		pass

	@abstractmethod
	async def fetch(self, ids: List[str]) -> List[List[float]]:
		"""
		Fetch the embeddings of the ids.
		"""
		pass

	@abstractmethod
	async def is_exist(self, ids: List[str]) -> List[bool]:
		"""
		Check if the ids exist in the Vector DB.
		"""
		pass

	@abstractmethod
	async def delete(self, ids: List[str]):
		pass

	def truncated_inputs(self, inputs: List[str]) -> List[str]:
		if isinstance(self.embedding, OpenAIEmbedding):
			openai_embedding_limit = 8000
			results = openai_truncate_by_token(
				inputs, openai_embedding_limit, self.embedding.model_name
			)
			return results
		return inputs
