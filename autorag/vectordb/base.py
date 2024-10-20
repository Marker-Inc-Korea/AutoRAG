from abc import abstractmethod
from typing import List, Tuple

from autorag import embedding_models


class BaseVectorStore:
	support_similarity_metrics = ["l2", "ip", "cosine"]

	def __init__(self, embedding_model: str, similarity_metric: str = "cosine"):
		self.embedding = embedding_models[embedding_model]()
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
	async def delete(self, ids: List[str]):
		pass