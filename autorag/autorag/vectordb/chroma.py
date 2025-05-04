from typing import List, Optional, Dict, Tuple, Union

from chromadb import (
	EphemeralClient,
	PersistentClient,
	DEFAULT_TENANT,
	DEFAULT_DATABASE,
	CloudClient,
	AsyncHttpClient,
)
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import QueryResult

from autorag.utils.util import apply_recursive
from autorag.vectordb.base import BaseVectorStore


class Chroma(BaseVectorStore):
	def __init__(
		self,
		embedding_model: Union[str, List[dict]],
		collection_name: str,
		embedding_batch: int = 100,
		client_type: str = "persistent",
		similarity_metric: str = "cosine",
		path: str = None,
		host: str = "localhost",
		port: int = 8000,
		ssl: bool = False,
		headers: Optional[Dict[str, str]] = None,
		api_key: Optional[str] = None,
		tenant: str = DEFAULT_TENANT,
		database: str = DEFAULT_DATABASE,
	):
		super().__init__(embedding_model, similarity_metric, embedding_batch)
		if client_type == "ephemeral":
			self.client = EphemeralClient(tenant=tenant, database=database)
		elif client_type == "persistent":
			assert path is not None, "path must be provided for persistent client"
			self.client = PersistentClient(path=path, tenant=tenant, database=database)
		elif client_type == "http":
			self.client = AsyncHttpClient(
				host=host,
				port=port,
				ssl=ssl,
				headers=headers,
				tenant=tenant,
				database=database,
			)
		elif client_type == "cloud":
			self.client = CloudClient(
				tenant=tenant,
				database=database,
				api_key=api_key,
			)
		else:
			raise ValueError(
				f"client_type {client_type} is not supported\n"
				"supported client types are: ephemeral, persistent, http, cloud"
			)

		self.collection = self.client.get_or_create_collection(
			name=collection_name,
			metadata={"hnsw:space": similarity_metric},
		)

	async def add(self, ids: List[str], texts: List[str]):
		texts = self.truncated_inputs(texts)
		text_embeddings = await self.embedding.aget_text_embedding_batch(texts)
		if isinstance(self.collection, AsyncCollection):
			await self.collection.add(ids=ids, embeddings=text_embeddings)
		else:
			self.collection.add(ids=ids, embeddings=text_embeddings)

	async def fetch(self, ids: List[str]) -> List[List[float]]:
		if isinstance(self.collection, AsyncCollection):
			fetch_result = await self.collection.get(ids, include=["embeddings"])
		else:
			fetch_result = self.collection.get(ids, include=["embeddings"])
		fetch_embeddings = fetch_result["embeddings"]
		return fetch_embeddings

	async def is_exist(self, ids: List[str]) -> List[bool]:
		if isinstance(self.collection, AsyncCollection):
			fetched_result = await self.collection.get(ids, include=[])
		else:
			fetched_result = self.collection.get(ids, include=[])
		existed_ids = fetched_result["ids"]
		return list(map(lambda x: x in existed_ids, ids))

	async def query(
		self, queries: List[str], top_k: int, **kwargs
	) -> Tuple[List[List[str]], List[List[float]]]:
		queries = self.truncated_inputs(queries)
		query_embeddings: List[
			List[float]
		] = await self.embedding.aget_text_embedding_batch(queries)
		if isinstance(self.collection, AsyncCollection):
			query_result: QueryResult = await self.collection.query(
				query_embeddings=query_embeddings, n_results=top_k
			)
		else:
			query_result: QueryResult = self.collection.query(
				query_embeddings=query_embeddings, n_results=top_k
			)
		ids = query_result["ids"]
		scores = query_result["distances"]
		scores = apply_recursive(lambda x: 1 - x, scores)
		return ids, scores

	async def delete(self, ids: List[str]):
		if isinstance(self.collection, AsyncCollection):
			await self.collection.delete(ids)
		else:
			self.collection.delete(ids)
