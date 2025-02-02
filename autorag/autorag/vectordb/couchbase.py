import logging

from datetime import timedelta

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions

from typing import List, Tuple, Optional, Union

from autorag.utils.util import make_batch
from autorag.vectordb import BaseVectorStore

logger = logging.getLogger("AutoRAG")


class Couchbase(BaseVectorStore):
	def __init__(
		self,
		embedding_model: Union[str, List[dict]],
		bucket_name: str,
		scope_name: str,
		collection_name: str,
		index_name: str,
		embedding_batch: int = 100,
		connection_string: str = "",
		username: str = "",
		password: str = "",
		ingest_batch: int = 100,
		text_key: Optional[str] = "text",
		embedding_key: Optional[str] = "embedding",
		scoped_index: bool = True,
	):
		super().__init__(
			embedding_model=embedding_model,
			similarity_metric="ip",
			embedding_batch=embedding_batch,
		)

		self.index_name = index_name
		self.bucket_name = bucket_name
		self.scope_name = scope_name
		self.collection_name = collection_name
		self.scoped_index = scoped_index
		self.text_key = text_key
		self.embedding_key = embedding_key
		self.ingest_batch = ingest_batch

		auth = PasswordAuthenticator(username, password)
		self.cluster = Cluster(connection_string, ClusterOptions(auth))

		# Wait until the cluster is ready for use.
		self.cluster.wait_until_ready(timedelta(seconds=5))

		# Check if the bucket exists
		if not self._check_bucket_exists():
			raise ValueError(
				f"Bucket {self.bucket_name} does not exist. "
				" Please create the bucket before searching."
			)

		try:
			self.bucket = self.cluster.bucket(self.bucket_name)
			self.scope = self.bucket.scope(self.scope_name)
			self.collection = self.scope.collection(self.collection_name)
		except Exception as e:
			raise ValueError(
				"Error connecting to couchbase. "
				"Please check the connection and credentials."
			) from e

		# Check if the index exists. Throws ValueError if it doesn't
		try:
			self._check_index_exists()
		except Exception:
			raise

		# Reinitialize to ensure a consistent state
		self.bucket = self.cluster.bucket(self.bucket_name)
		self.scope = self.bucket.scope(self.scope_name)
		self.collection = self.scope.collection(self.collection_name)

	async def add(self, ids: List[str], texts: List[str]):
		from couchbase.exceptions import DocumentExistsException

		texts = self.truncated_inputs(texts)
		text_embeddings: List[
			List[float]
		] = await self.embedding.aget_text_embedding_batch(texts)

		documents_to_insert = []
		for _id, text, embedding in zip(ids, texts, text_embeddings):
			doc = {
				self.text_key: text,
				self.embedding_key: embedding,
			}
			documents_to_insert.append({_id: doc})

		batch_documents_to_insert = make_batch(documents_to_insert, self.ingest_batch)

		for batch in batch_documents_to_insert:
			insert_batch = {}
			for doc in batch:
				insert_batch.update(doc)
			try:
				self.collection.upsert_multi(insert_batch)
			except DocumentExistsException as e:
				logger.debug(f"Document already exists: {e}")

	async def fetch(self, ids: List[str]) -> List[List[float]]:
		# Fetch vectors by IDs
		fetched_result = self.collection.get_multi(ids)
		fetched_vectors = {
			k: v.value[f"{self.embedding_key}"]
			for k, v in fetched_result.results.items()
		}
		return list(map(lambda x: fetched_vectors[x], ids))

	async def is_exist(self, ids: List[str]) -> List[bool]:
		existed_result = self.collection.exists_multi(ids)
		existed_ids = {k: v.exists for k, v in existed_result.results.items()}
		return list(map(lambda x: existed_ids[x], ids))

	async def query(
		self, queries: List[str], top_k: int, **kwargs
	) -> Tuple[List[List[str]], List[List[float]]]:
		import couchbase.search as search
		from couchbase.options import SearchOptions
		from couchbase.vector_search import VectorQuery, VectorSearch

		queries = self.truncated_inputs(queries)
		query_embeddings: List[
			List[float]
		] = await self.embedding.aget_text_embedding_batch(queries)

		ids, scores = [], []
		for query_embedding in query_embeddings:
			# Create Search Request
			search_req = search.SearchRequest.create(
				VectorSearch.from_vector_query(
					VectorQuery(
						self.embedding_key,
						query_embedding,
						top_k,
					)
				)
			)

			# Search
			if self.scoped_index:
				search_iter = self.scope.search(
					self.index_name,
					search_req,
					SearchOptions(limit=top_k),
				)

			else:
				search_iter = self.cluster.search(
					self.index_name,
					search_req,
					SearchOptions(limit=top_k),
				)

			# Parse the search results
			# search_iter.rows() can only be iterated once.
			id_list, score_list = [], []
			for result in search_iter.rows():
				id_list.append(result.id)
				score_list.append(result.score)

			ids.append(id_list)
			scores.append(score_list)

		return ids, scores

	async def delete(self, ids: List[str]):
		self.collection.remove_multi(ids)

	def _check_bucket_exists(self) -> bool:
		"""Check if the bucket exists in the linked Couchbase cluster.

		Returns:
		    True if the bucket exists
		"""
		bucket_manager = self.cluster.buckets()
		try:
			bucket_manager.get_bucket(self.bucket_name)
			return True
		except Exception as e:
			logger.debug("Error checking if bucket exists:", e)
			return False

	def _check_index_exists(self) -> bool:
		"""Check if the Search index exists in the linked Couchbase cluster
		Returns:
		    bool: True if the index exists, False otherwise.
		    Raises a ValueError if the index does not exist.
		"""
		if self.scoped_index:
			all_indexes = [
				index.name for index in self.scope.search_indexes().get_all_indexes()
			]
			if self.index_name not in all_indexes:
				raise ValueError(
					f"Index {self.index_name} does not exist. "
					" Please create the index before searching."
				)
		else:
			all_indexes = [
				index.name for index in self.cluster.search_indexes().get_all_indexes()
			]
			if self.index_name not in all_indexes:
				raise ValueError(
					f"Index {self.index_name} does not exist. "
					" Please create the index before searching."
				)

		return True
