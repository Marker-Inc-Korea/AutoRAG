from typing import Optional
from pydantic import BaseModel

import os
import logging

# Set log level based on environment variable
log_level = logging.DEBUG if os.getenv("AUTORAG_ENV") == "dev" else logging.WARNING
logger = logging.getLogger(__name__)
logger.setLevel(log_level)


class Project(BaseModel):
	id: str
	name: str
	description: Optional[str] = None
	_client: Optional[object] = None  # Reference to the client instance

	def __init__(self, **data):
		super().__init__(**data)
		self._client = None

	def set_client(self, client):
		self._client = client

	async def upload_file(self, file_pattern: str):
		"""Upload files to the project"""
		if not self._client:
			raise RuntimeError("Project is not associated with a client")
		return await self._client.upload_file(self.id, file_pattern)

	async def embedding(self, vector_storage: str = "auto"):
		"""Create embeddings for project files"""
		if not self._client:
			raise RuntimeError("Project is not associated with a client")
		return await self._client.embedding(self.id, vector_storage)

	async def create_rag_pipeline(self, embedding_model: str = "auto") -> "RAGPipeline":
		"""Create a RAG pipeline for this project"""
		if not self._client:
			raise RuntimeError("Project is not associated with a client")
		return await self._client.create_rag_pipeline(self.id, embedding_model)


class Passage(BaseModel):
	doc_id: str
	content: str
	score: float


class RetrievedPassage(BaseModel):
	content: str
	doc_id: str
	filepath: Optional[str] = None
	file_page: Optional[int] = None
	start_idx: Optional[int] = None
	end_idx: Optional[int] = None


class Retrieval(BaseModel):
	text: str
	score: float
	metadata: dict


class RetrievalResults:
	def __init__(self, retrievals: list, **kwargs):
		# Convert dict to Retrieval objects if needed
		self.retrievals = [
			r if isinstance(r, Retrieval) else Retrieval(**r) for r in retrievals
		]
		for key, value in kwargs.items():
			setattr(self, key, value)

	def to_prompt_string(self):
		"""Convert retrievals to a string format for prompts"""
		context_parts = []
		for i, retrieval in enumerate(self.retrievals, 1):
			source = retrieval.metadata.get("source", "unknown")
			context_parts.append(f"[{i}] Source: {source}\n{retrieval.text}\n")
		return "\n".join(context_parts)

	def __str__(self) -> str:
		"""String representation of retrievals"""
		return "\n".join(
			f"- {r.text} (score: {r.score}, source: {r.metadata.get('doc_id', 'unknown doc')}::chunk_{r.metadata.get('chunk_id', 'unknown chunk_id')})"
			for r in self.retrievals
		)


class RAGPipeline:
	def __init__(self, client, project_id: str):
		self.client = client
		self.project_id = project_id

	async def evaluate(self) -> dict:
		"""Evaluate RAG pipeline"""
		session = await self.client._ensure_session()
		try:
			response = await self.client._post(f"/projects/{self.project_id}/evaluate")

			return response
		except Exception as e:
			logger.error(f"Error evaluating RAG pipeline: {str(e)}")
			# 서버 오류 시 기본 평가 결과 반환
			return {
				"overall_metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
				"retriever_metrics": {"claim_recall": 0.0, "context_precision": 0.0},
				"generator_metrics": {
					"context_utilization": 0.0,
					"noise_sensitivity_in_relevant": 0.0,
					"noise_sensitivity_in_irrelevant": 0.0,
					"hallucination": 0.0,
					"self_knowledge": 0.0,
					"faithfulness": 0.0,
				},
			}

		finally:
			if session and not session.closed:
				await session.close()
				self.session = None
