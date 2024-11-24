import logging

from pymongo import MongoClient


from typing import List, Optional, Tuple

from autorag.utils.util import apply_recursive
from autorag.vectordb import BaseVectorStore

logger = logging.getLogger("AutoRAG")


class MongoDBAtlas(BaseVectorStore):
	def __init__(
		self,
		embedding_model: str,
		embedding_batch: int = 100,
		similarity_metric: str = "cosine",
		uri: str = "http://localhost:19530",
		db_name: str = "",
		collection_name: str = "",
		index_name: str = "",
		token: str = "",
		user: str = "",
		password: str = "",
		timeout: Optional[float] = None,
	):
		super().__init__(embedding_model, similarity_metric, embedding_batch)

		self.client = MongoClient(uri)

	async def add(self, ids: List[str], texts: List[str]):
		texts = self.truncated_inputs(texts)
		text_embeddings: List[
			List[float]
		] = await self.embedding.aget_text_embedding_batch(texts)

		# make data for insertion
		data = list(
			map(lambda _id, vector: {"id": _id, "vector": vector}, ids, text_embeddings)
		)

		# Insert data into the collection
		res = self.collection.insert(data=data, timeout=self.timeout)
		assert (
			res.insert_count == len(ids)
		), f"Insertion failed. Try to insert {len(ids)} but only {res['insert_count']} inserted."

		self.collection.flush(timeout=self.timeout)

		index_params = {
			"index_type": "IVF_FLAT",
			"metric_type": self.similarity_metric.upper(),
			"params": {},
		}  # TODO : add params
		self.collection.create_index(
			field_name="vector", index_params=index_params, timeout=self.timeout
		)

	async def query(
		self, queries: List[str], top_k: int, **kwargs
	) -> Tuple[List[List[str]], List[List[float]]]:
		queries = self.truncated_inputs(queries)
		query_embeddings: List[
			List[float]
		] = await self.embedding.aget_text_embedding_batch(queries)

		self.collection.load(timeout=self.timeout)

		# Perform similarity search
		results = self.collection.search(
			data=query_embeddings,
			limit=top_k,
			anns_field="vector",
			param={"metric_type": self.similarity_metric.upper()},
			timeout=self.timeout,
			**kwargs,
		)

		# Extract IDs and distances
		ids = [[str(hit.id) for hit in result] for result in results]
		distances = [[hit.distance for hit in result] for result in results]

		if self.similarity_metric in ["l2"]:
			distances = apply_recursive(lambda x: -x, distances)

		return ids, distances

	async def fetch(self, ids: List[str]) -> List[List[float]]:
		try:
			self.collection.load(timeout=self.timeout)
		except MilvusException as e:
			logger.warning(f"Failed to load collection: {e}")
			return [[]] * len(ids)
		# Fetch vectors by IDs
		results = self.collection.query(
			expr=f"id in {ids}", output_fields=["id", "vector"], timeout=self.timeout
		)
		id_vector_dict = {str(result["id"]): result["vector"] for result in results}
		result = [id_vector_dict[_id] for _id in ids]
		return result

	async def is_exist(self, ids: List[str]) -> List[bool]:
		try:
			self.collection.load(timeout=self.timeout)
		except MilvusException:
			return [False] * len(ids)
		# Check the existence of IDs
		results = self.collection.query(
			expr=f"id in {ids}", output_fields=["id"], timeout=self.timeout
		)
		# Determine existence
		existing_ids = {str(result["id"]) for result in results}
		return [str(_id) in existing_ids for _id in ids]

	async def delete(self, ids: List[str]):
		# Delete entries by IDs
		self.collection.delete(expr=f"id in {ids}", timeout=self.timeout)

	def delete_collection(self):
		# Delete the collection
		self.collection.release(timeout=self.timeout)
		self.collection.drop_index(timeout=self.timeout)
		self.collection.drop(timeout=self.timeout)
