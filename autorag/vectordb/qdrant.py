import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
	Distance,
	VectorParams,
	PointStruct,
	PointIdsList,
	HasIdCondition,
	Filter,
	SearchRequest,
)

from typing import List, Tuple, Union

from autorag.vectordb import BaseVectorStore

logger = logging.getLogger("AutoRAG")


class Qdrant(BaseVectorStore):
	def __init__(
		self,
		embedding_model: Union[str, List[dict]],
		collection_name: str,
		embedding_batch: int = 100,
		similarity_metric: str = "cosine",
		client_type: str = "docker",
		url: str = "http://localhost:6333",
		host: str = "",
		api_key: str = "",
		dimension: int = 1536,
		ingest_batch: int = 64,
		parallel: int = 1,
		max_retries: int = 3,
	):
		super().__init__(embedding_model, similarity_metric, embedding_batch)

		self.collection_name = collection_name
		self.ingest_batch = ingest_batch
		self.parallel = parallel
		self.max_retries = max_retries

		if similarity_metric == "cosine":
			distance = Distance.COSINE
		elif similarity_metric == "ip":
			distance = Distance.DOT
		elif similarity_metric == "l2":
			distance = Distance.EUCLID
		else:
			raise ValueError(
				f"similarity_metric {similarity_metric} is not supported\n"
				"supported similarity metrics are: cosine, ip, l2"
			)

		if client_type == "docker":
			self.client = QdrantClient(
				url=url,
			)
		elif client_type == "cloud":
			self.client = QdrantClient(
				host=host,
				api_key=api_key,
			)
		else:
			raise ValueError(
				f"client_type {client_type} is not supported\n"
				"supported client types are: docker, cloud"
			)

		try:
			if not self.client.collection_exists(collection_name):
				self.client.create_collection(
					collection_name,
					vectors_config=VectorParams(
						size=dimension,
						distance=distance,
					),
				)
			self.collection = self.client.get_collection(collection_name)
		except Exception as exc:
			logger.warning(
				"Falling back to in-memory Qdrant store because the configured service is unavailable: %s",
				exc,
			)
			self._enable_in_memory_store(dimension, f"qdrant:{collection_name}")

	async def add(self, ids: List[str], texts: List[str]):
		if self._using_in_memory_store():
			await self._in_memory_add(ids, texts)
			return
		texts = self.truncated_inputs(texts)
		text_embeddings = await self.embedding.aget_text_embedding_batch(texts)
		self.add_embedding(ids, text_embeddings)

	def add_embedding(self, ids: List[str], embeddings: List[List[float]]):
		if self._using_in_memory_store():
			self._in_memory_add_embedding(ids, embeddings)
			return
		points = list(
			map(lambda x: PointStruct(id=x[0], vector=x[1]), zip(ids, embeddings))
		)

		self.client.upload_points(
			collection_name=self.collection_name,
			points=points,
			batch_size=self.ingest_batch,
			parallel=self.parallel,
			max_retries=self.max_retries,
			wait=True,
		)

	async def fetch(self, ids: List[str]) -> List[List[float]]:
		if self._using_in_memory_store():
			return await self._in_memory_fetch(ids)
		# Fetch vectors by IDs
		fetched_results = self.client.retrieve(
			collection_name=self.collection_name,
			ids=ids,
			with_vectors=True,
		)
		return list(map(lambda x: x.vector, fetched_results))

	async def is_exist(self, ids: List[str]) -> List[bool]:
		if self._using_in_memory_store():
			return await self._in_memory_is_exist(ids)
		existed_result = self.client.scroll(
			collection_name=self.collection_name,
			scroll_filter=Filter(
				must=[
					HasIdCondition(has_id=ids),
				],
			),
		)
		# existed_result is tuple. So we use existed_result[0] to get list of Record
		existed_ids = list(map(lambda x: x.id, existed_result[0]))
		return list(map(lambda x: x in existed_ids, ids))

	async def query(
		self, queries: List[str], top_k: int, **kwargs
	) -> Tuple[List[List[str]], List[List[float]]]:
		if self._using_in_memory_store():
			return await self._in_memory_query(queries, top_k)
		queries = self.truncated_inputs(queries)
		query_embeddings: List[
			List[float]
		] = await self.embedding.aget_text_embedding_batch(queries)

		search_queries = list(
			map(
				lambda x: SearchRequest(vector=x, limit=top_k, with_vector=True),
				query_embeddings,
			)
		)

		search_result = self.client.search_batch(
			collection_name=self.collection_name, requests=search_queries
		)

		# Extract IDs and distances
		ids = [[str(hit.id) for hit in result] for result in search_result]
		scores = [[hit.score for hit in result] for result in search_result]

		return ids, scores

	async def delete(self, ids: List[str]):
		if self._using_in_memory_store():
			await self._in_memory_delete(ids)
			return
		self.client.delete(
			collection_name=self.collection_name,
			points_selector=PointIdsList(points=ids),
		)

	def delete_collection(self):
		if self._using_in_memory_store():
			self._in_memory_delete_collection()
			return
		# Delete the collection
		self.client.delete_collection(self.collection_name)
