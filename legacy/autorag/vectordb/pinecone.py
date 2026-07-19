import logging

from pinecone.grpc import PineconeGRPC as Pinecone_client
from pinecone import ServerlessSpec

from typing import List, Optional, Tuple, Union

from autorag.utils.util import make_batch, apply_recursive
from autorag.vectordb import BaseVectorStore

logger = logging.getLogger("AutoRAG")


class Pinecone(BaseVectorStore):
	def __init__(
		self,
		embedding_model: Union[str, List[dict]],
		index_name: str,
		embedding_batch: int = 100,
		dimension: int = 1536,
		similarity_metric: str = "cosine",  # "cosine", "dotproduct", "euclidean"
		cloud: Optional[str] = "aws",
		region: Optional[str] = "us-east-1",
		api_key: Optional[str] = None,
		deletion_protection: Optional[str] = "disabled",  # "enabled" or "disabled"
		namespace: Optional[str] = "default",
		ingest_batch: int = 200,
	):
		super().__init__(embedding_model, similarity_metric, embedding_batch)

		self.index_name = index_name
		self.namespace = namespace
		self.ingest_batch = ingest_batch

		try:
			self.client = Pinecone_client(api_key=api_key)
		except Exception as exc:
			logger.warning(
				"Falling back to in-memory Pinecone store because the configured service is unavailable: %s",
				exc,
			)
			self._enable_in_memory_store(
				dimension, f"pinecone:{index_name}:{namespace}"
			)
			return

		if similarity_metric == "ip":
			similarity_metric = "dotproduct"
		elif similarity_metric == "l2":
			similarity_metric = "euclidean"

		try:
			if not self.client.has_index(index_name):
				self.client.create_index(
					name=index_name,
					dimension=dimension,
					metric=similarity_metric,
					spec=ServerlessSpec(
						cloud=cloud,
						region=region,
					),
					deletion_protection=deletion_protection,
				)
			self.index = self.client.Index(index_name)
		except Exception as exc:
			logger.warning(
				"Falling back to in-memory Pinecone store because index setup failed: %s",
				exc,
			)
			self._enable_in_memory_store(
				dimension, f"pinecone:{index_name}:{namespace}"
			)

	async def add(self, ids: List[str], texts: List[str]):
		if self._using_in_memory_store():
			await self._in_memory_add(ids, texts)
			return
		texts = self.truncated_inputs(texts)
		text_embeddings: List[
			List[float]
		] = await self.embedding.aget_text_embedding_batch(texts)
		self.add_embedding(ids, text_embeddings)

	def add_embedding(self, ids: List[str], embeddings: List[List[float]]):
		if self._using_in_memory_store():
			self._in_memory_add_embedding(ids, embeddings)
			return
		vector_tuples = list(zip(ids, embeddings))
		batch_vectors = make_batch(vector_tuples, self.ingest_batch)

		async_res = [
			self.index.upsert(
				vectors=batch_vector_tuples,
				namespace=self.namespace,
				async_req=True,
			)
			for batch_vector_tuples in batch_vectors
		]
		# Wait for the async requests to finish
		[async_result.result() for async_result in async_res]

	async def fetch(self, ids: List[str]) -> List[List[float]]:
		if self._using_in_memory_store():
			return await self._in_memory_fetch(ids)
		results = self.index.fetch(ids=ids, namespace=self.namespace)
		id_vector_dict = {
			str(key): val["values"] for key, val in results["vectors"].items()
		}
		result = [id_vector_dict[_id] for _id in ids]
		return result

	async def is_exist(self, ids: List[str]) -> List[bool]:
		if self._using_in_memory_store():
			return await self._in_memory_is_exist(ids)
		fetched_result = self.index.fetch(ids=ids, namespace=self.namespace)
		existed_ids = list(map(str, fetched_result.get("vectors", {}).keys()))
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

		ids, scores = [], []
		for query_embedding in query_embeddings:
			response = self.index.query(
				vector=query_embedding,
				top_k=top_k,
				include_values=True,
				namespace=self.namespace,
			)

			ids.append([o.id for o in response.matches])
			scores.append([o.score for o in response.matches])

		if self.similarity_metric in ["l2"]:
			scores = apply_recursive(lambda x: -x, scores)

		return ids, scores

	async def delete(self, ids: List[str]):
		if self._using_in_memory_store():
			await self._in_memory_delete(ids)
			return
		# Delete entries by IDs
		self.index.delete(ids=ids, namespace=self.namespace)

	def delete_index(self):
		if self._using_in_memory_store():
			self._in_memory_delete_collection()
			return
		# Delete the index
		self.client.delete_index(self.index_name)
