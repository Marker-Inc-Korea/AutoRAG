from typing import Optional
import aiohttp
import os
import glob
from .exceptions import APIError
from .models import Project, RAGPipeline, RetrievalResults
import logging

logger = logging.getLogger("AutoRAG-Client")


class AutoRAGClient:
	def __init__(
		self,
		api_key: str,
		base_url: str = os.getenv("AUTORAG_BASE_URL", "https://raas.auto-rag.com"),
	):
		self.token = api_key
		self.base_url = base_url.rstrip("/")
		self.headers = {
			"Authorization": f"Bearer {api_key}",
			"Content-Type": "application/json",
		}
		self._projects = {}
		self._files = {}
		self.session = None

	async def _ensure_session(self):
		"""Ensure we have an active session"""
		if self.session is None or self.session.closed:
			self.session = aiohttp.ClientSession(headers=self.headers)
		return self.session

	async def _post(self, endpoint: str, json: dict = None) -> dict:
		"""Internal method for making POST requests"""
		session = await self._ensure_session()
		async with session.post(f"{self.base_url}{endpoint}", json=json) as response:
			if response.status >= 400:
				raise APIError(f"API request failed with status {response.status}")
			return await response.json()

	async def create_project(
		self, name: str, description: Optional[str] = None
	) -> Project:
		"""Create a new project"""
		payload = {
			"name": name,
		}
		if description:
			payload["description"] = description

		try:
			response = await self._post("/projects", payload)
			project_id = response.get("id")

			if not project_id:
				raise APIError("Failed to retrieve project ID from response")

			project = Project(
				id=project_id,
				name=response.get("name"),
				description=response.get("description"),
			)
			project.set_client(self)
			self._projects[project_id] = project
			self._files[project_id] = []
			return project
		except APIError as e:
			raise APIError(f"Failed to create project: {str(e)}")

	async def upload_file(self, project_id: str, file_pattern: str):
		"""Upload files to the project"""
		if project_id not in self._projects:
			raise ValueError(f"Project {project_id} does not exist")

		# Print current working directory
		current_dir = os.getcwd()
		logger.debug(f"Current working directory: {current_dir}")

		# Split directory and file pattern
		dir_path = os.path.dirname(file_pattern)
		file_glob = os.path.basename(file_pattern)

		# Convert file pattern (e.g., *.[pdf|txt|csv|md] -> *.{pdf,txt,csv,md})
		if file_glob.startswith("*.[") and file_glob.endswith("]"):
			extensions = file_glob[3:-1].replace("|", ",")
			file_glob = f"*.{{{extensions}}}"

		# Create new search pattern
		search_pattern = os.path.join(dir_path, file_glob)
		logger.debug(f"Converted pattern: {search_pattern}")

		# Search for files
		files = glob.glob(search_pattern, recursive=True)
		logger.debug(f"Found files: {files}")

		if not files:
			logger.warning(f"No files found matching pattern: {search_pattern}")
			return

		uploaded_files = []
		for file_path in files:
			try:
				file_data = {
					"path": file_path,
				}
				uploaded_files.append(file_data)
				self._files[project_id].append(file_data)
				logger.info(f"Successfully read file: {file_path}")

			except Exception as e:
				logger.error(f"Could not read file {file_path}: {str(e)}")

		if uploaded_files:
			await self._post(f"/projects/{project_id}/files", {"files": uploaded_files})
			logger.info(f"Successfully uploaded {len(uploaded_files)} files")
		else:
			logger.warning("No files were successfully uploaded")

	async def embedding(self, project_id: str, vector_storage: str = "auto"):
		"""Create embeddings for project files"""
		if project_id not in self._projects:
			raise APIError("Project not found")
		if not self._files[project_id]:
			raise APIError("No files uploaded to embed")

		# TODO: Change this
		await self._post(
			f"/projects/{project_id}/embedding", {"vector_storage": vector_storage}
		)

	async def create_rag_pipeline(
		self, project_id: str, embedding_model: str = "auto"
	) -> RAGPipeline:
		"""Create a RAG pipeline"""
		if project_id not in self._projects:
			raise APIError(f"Project {project_id} not found")

		try:
			await self._post(
				f"/projects/{project_id}/pipeline", {"embedding_model": embedding_model}
			)

			pipeline = RAGPipeline(self, project_id)
			return pipeline
		except Exception as e:
			logger.error(f"Error creating RAG pipeline: {str(e)}")
			# Even if API fails, return pipeline for mock functionality
			return RAGPipeline(self, project_id)

	async def __aenter__(self):
		"""Enter the runtime context related to this object."""
		self.session = await self._ensure_session()
		return self

	async def __aexit__(self, exc_type, exc, tb):
		"""Exit the runtime context related to this object."""
		if self.session and not self.session.closed:
			await self.session.close()
			self.session = None

	async def get_retrievals(self, rag_pipeline, question: str) -> RetrievalResults:
		"""Get retrievals for a query"""
		if not question:
			raise ValueError("question cannot be empty")
		response = await self._post(
			f"/projects/{rag_pipeline.project_id}/rag_contexts",
			json={"question": question},
		)
		return RetrievalResults(**response)
