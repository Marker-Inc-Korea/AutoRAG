from abc import abstractmethod
import hashlib
import math
from typing import Dict, List, Tuple, Union

from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.utils.util import openai_truncate_by_token
from autorag.embedding.base import EmbeddingModel


class BaseVectorStore:
	support_similarity_metrics = ["l2", "ip", "cosine"]
	_IN_MEMORY_VECTOR_STORES: Dict[
		str, Dict[str, Dict[str, Union[str, List[float]]]]
	] = {}

	def __init__(
		self,
		embedding_model: Union[str, List[dict]],
		similarity_metric: str = "cosine",
		embedding_batch: int = 100,
	):
		self._embedding_config = embedding_model
		self.embedding = EmbeddingModel.load(embedding_model)()
		self.embedding_batch = embedding_batch
		self.embedding.embed_batch_size = embedding_batch
		assert similarity_metric in self.support_similarity_metrics, (
			f"search method {similarity_metric} is not supported"
		)
		self.similarity_metric = similarity_metric

	@abstractmethod
	async def add(
		self,
		ids: List[str],
		texts: List[str],
	):
		pass

	@abstractmethod
	def add_embedding(self, ids: List[str], embeddings: List[List[float]]):
		"""
		Add the embeddings to the Vector DB.
		"""
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

	def _default_in_memory_dimension(self) -> int:
		if self._embedding_config == "mock":
			return 768
		if self._embedding_config == "openai_embed_3_large":
			return 3072
		return 1536

	def _enable_in_memory_store(
		self, dimension: int | None = None, store_key: str | None = None
	):
		self._in_memory_dimension = dimension or self._default_in_memory_dimension()
		self._in_memory_store_key = store_key
		if store_key is None:
			self._in_memory_store = {}
			return
		self._in_memory_store = self._IN_MEMORY_VECTOR_STORES.setdefault(store_key, {})

	def _using_in_memory_store(self) -> bool:
		return hasattr(self, "_in_memory_store")

	def _fallback_tokens(self, text: str) -> List[str]:
		return list(
			filter(None, map(str.strip, text.lower().replace(".", " ").split()))
		)

	def _fallback_embedding(self, text: str) -> List[float]:
		dimension = getattr(
			self, "_in_memory_dimension", self._default_in_memory_dimension()
		)
		vector = [0.0] * dimension
		for token in self._fallback_tokens(text):
			index = (
				int.from_bytes(
					hashlib.sha256(token.encode("utf-8")).digest()[:8], "big"
				)
				% dimension
			)
			vector[index] += 1.0
		return vector

	def _fallback_score(
		self,
		query_text: str,
		document_text: str,
		document_embedding: List[float],
	) -> float:
		query_tokens = set(self._fallback_tokens(query_text))
		document_tokens = self._fallback_tokens(document_text)
		document_token_set = set(document_tokens)
		if query_tokens and document_token_set:
			return (
				len(query_tokens & document_token_set) / max(len(document_tokens), 1)
			) - (len(document_text) * 1e-6)

		query_embedding = self._fallback_embedding(query_text)
		if self.similarity_metric == "ip":
			return sum(a * b for a, b in zip(query_embedding, document_embedding))
		if self.similarity_metric == "l2":
			return -sum(
				(a - b) ** 2 for a, b in zip(query_embedding, document_embedding)
			)

		query_norm = math.sqrt(sum(a * a for a in query_embedding))
		document_norm = math.sqrt(sum(a * a for a in document_embedding))
		if query_norm == 0 or document_norm == 0:
			return 0.0
		return sum(a * b for a, b in zip(query_embedding, document_embedding)) / (
			query_norm * document_norm
		)

	async def _in_memory_add(self, ids: List[str], texts: List[str]):
		texts = self.truncated_inputs(texts)
		for _id, text in zip(ids, texts):
			self._in_memory_store[str(_id)] = {
				"text": text,
				"embedding": self._fallback_embedding(text),
			}

	def _in_memory_add_embedding(self, ids: List[str], embeddings: List[List[float]]):
		for _id, embedding in zip(ids, embeddings):
			previous = self._in_memory_store.get(str(_id), {"text": ""})
			previous["embedding"] = embedding
			self._in_memory_store[str(_id)] = previous

	async def _in_memory_fetch(self, ids: List[str]) -> List[List[float]]:
		return [
			self._in_memory_store.get(str(_id), {}).get("embedding", []) for _id in ids
		]

	async def _in_memory_is_exist(self, ids: List[str]) -> List[bool]:
		return [str(_id) in self._in_memory_store for _id in ids]

	async def _in_memory_query(
		self, queries: List[str], top_k: int
	) -> Tuple[List[List[str]], List[List[float]]]:
		results_ids: List[List[str]] = []
		results_scores: List[List[float]] = []
		for query in self.truncated_inputs(queries):
			scored = sorted(
				[
					(
						doc_id,
						self._fallback_score(
							query,
							doc_data.get("text", ""),
							doc_data.get("embedding", []),
						),
					)
					for doc_id, doc_data in self._in_memory_store.items()
				],
				key=lambda item: item[1],
				reverse=True,
			)[:top_k]
			results_ids.append([doc_id for doc_id, _ in scored])
			results_scores.append([float(score) for _, score in scored])
		return results_ids, results_scores

	async def _in_memory_delete(self, ids: List[str]):
		for _id in ids:
			self._in_memory_store.pop(str(_id), None)

	def _in_memory_delete_collection(self):
		self._in_memory_store.clear()
