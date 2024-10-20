from typing import List, Optional, Dict, Sequence

from chromadb import (
	EphemeralClient,
	PersistentClient,
	DEFAULT_TENANT,
	DEFAULT_DATABASE,
	CloudClient,
	HttpClient,
)
from chromadb.api.types import IncludeEnum

from autorag.vectordb.base import BaseVectorStore
from langchain_chroma.vectorstores import Chroma as LangchainChroma


class Chroma(BaseVectorStore):
	support_algorithms = ["l2", "ip", "cosine"]

	def __init__(
		self,
		embedding_model: str,
		collection_name: str,
		client_type: str = "persistent",
		search_method: str = "cosine",
		path: str = None,
		host: str = "localhost",
		port: int = 8000,
		ssl: bool = False,
		headers: Optional[Dict[str, str]] = None,
		api_key: Optional[str] = None,
		tenant: str = DEFAULT_TENANT,
		database: str = DEFAULT_DATABASE,
	):
		super().__init__(embedding_model)
		if client_type == "ephemeral":
			client = EphemeralClient(tenant=tenant, database=database)
		elif client_type == "persistent":
			assert path is not None, "path must be provided for persistent client"
			client = PersistentClient(path=path, tenant=tenant, database=database)
		elif client_type == "http":
			client = HttpClient(
				host=host,
				port=port,
				ssl=ssl,
				headers=headers,
				tenant=tenant,
				database=database,
			)
		elif client_type == "cloud":
			client = CloudClient(
				tenant=tenant,
				database=database,
				api_key=api_key,
			)
		else:
			raise ValueError(
				f"client_type {client_type} is not supported\n"
				"supported client types are: ephemeral, persistent, http, cloud"
			)

		assert (
			search_method in self.support_algorithms
		), f"search method {search_method} is not supported"
		self.langchain_vector_store = LangchainChroma(
			collection_name=collection_name,
			collection_metadata={"hnsw:space": search_method},
			client=client,
			embedding_function=self.embedding_function,
			persist_directory=path,
		)

	def fetch(self, ids: Sequence[str]) -> List[str]:
		collection = self.langchain_vector_store._collection
		fetch_result = collection.get(ids, include=[IncludeEnum.documents])
		fetch_documents = fetch_result["documents"]
		return fetch_documents
