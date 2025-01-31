import logging
from typing import Any, Dict, List, Tuple, Optional, Union

from pymilvus import (
	DataType,
	FieldSchema,
	CollectionSchema,
	connections,
	Collection,
	MilvusException,
)
from pymilvus.orm import utility

from autorag.utils.util import apply_recursive
from autorag.vectordb import BaseVectorStore


logger = logging.getLogger("AutoRAG")


class Milvus(BaseVectorStore):
	def __init__(
		self,
		embedding_model: Union[str, List[dict]],
		collection_name: str,
		embedding_batch: int = 100,
		similarity_metric: str = "cosine",
		index_type: str = "IVF_FLAT",
		uri: str = "http://localhost:19530",
		db_name: str = "",
		token: str = "",
		user: str = "",
		password: str = "",
		timeout: Optional[float] = None,
		params: Dict[str, Any] = {},
	):
		super().__init__(embedding_model, similarity_metric, embedding_batch)

		# Connect to Milvus server
		connections.connect(
			"default",
			uri=uri,
			token=token,
			db_name=db_name,
			user=user,
			password=password,
		)
		self.collection_name = collection_name
		self.timeout = timeout
		self.params = params
		self.index_type = index_type

		# Set Collection
		if not utility.has_collection(collection_name, timeout=timeout):
			# Get the dimension of the embeddings
			test_embedding_result: List[float] = self.embedding.get_query_embedding(
				"test"
			)
			dimension = len(test_embedding_result)

			pk = FieldSchema(
				name="id",
				dtype=DataType.VARCHAR,
				max_length=128,
				is_primary=True,
				auto_id=False,
			)
			field = FieldSchema(
				name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension
			)
			schema = CollectionSchema(fields=[pk, field])

			self.collection = Collection(name=self.collection_name, schema=schema)
			index_params = {
				"metric_type": self.similarity_metric.upper(),
				"index_type": self.index_type.upper(),
				"params": self.params,
			}
			self.collection.create_index(
				field_name="vector", index_params=index_params, timeout=self.timeout
			)
		else:
			self.collection = Collection(name=self.collection_name)

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
