from typing import List, Tuple, Sequence

from langchain_core.embeddings import Embeddings
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag import embedding_models


class LlamaIndexEmbeddings(Embeddings):
	def __init__(self, llama_index_instance: BaseEmbedding):
		self.llama_index_instance = llama_index_instance

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		return self.llama_index_instance.get_text_embedding_batch(texts)

	def embed_query(self, text: str) -> list[float]:
		return self.llama_index_instance.get_text_embedding(text)

	async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
		return await self.llama_index_instance.aget_text_embedding_batch(texts)

	async def aembed_query(self, text: str) -> list[float]:
		return await self.llama_index_instance.aget_text_embedding(text)


class BaseVectorStore:
	def __init__(self, embedding_model: str):
		llama_index_embeddings = embedding_models[embedding_model]()
		self.embedding_function = LlamaIndexEmbeddings(llama_index_embeddings)
		self.langchain_vector_store = None

	def add(
		self,
		ids: List[str],
		texts: List[str],
	):
		self.langchain_vector_store.add_texts(texts, ids=ids)

	def query(
		self, queries: List[str], top_k: int, **kwargs
	) -> Tuple[List[List[str]], List[List[float]]]:
		content_result = []
		score_result = []
		for query in queries:
			retrieve_result = (
				self.langchain_vector_store.similarity_search_with_relevance_scores(
					query, k=top_k
				)
			)
			retrieved_contents = list(map(lambda x: x[0], retrieve_result))
			retrieved_scores = list(map(lambda x: x[1], retrieve_result))
			retrieved_contents = list(map(lambda x: x.page_content, retrieved_contents))
			content_result.append(retrieved_contents)
			score_result.append(retrieved_scores)
		return content_result, score_result

	def fetch(self, ids: Sequence[str]) -> List[str]:
		documents = self.langchain_vector_store.get_by_ids(ids)
		return list(map(lambda x: x.page_content, documents))

	def delete(self, ids: List[str]):
		self.langchain_vector_store.delete(ids)
