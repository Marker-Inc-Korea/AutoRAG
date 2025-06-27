import logging
import os
import pathlib
import uuid
from typing import Dict, Optional, List, Union, Literal

import pandas as pd
from quart import Quart, request, jsonify
from quart.helpers import stream_with_context
from pydantic import BaseModel, ValidationError

from autorag.deploy.base import BaseRunner
from autorag.nodes.generator.base import BaseGenerator
from autorag.nodes.promptmaker.base import BasePromptMaker
from autorag.utils.util import fetch_contents, to_list

logger = logging.getLogger("AutoRAG")

deploy_dir = pathlib.Path(__file__).parent
root_dir = pathlib.Path(__file__).parent.parent

VERSION_PATH = os.path.join(root_dir, "VERSION")


class QueryRequest(BaseModel):
	query: str
	result_column: Optional[str] = "generated_texts"


class RetrievedPassage(BaseModel):
	content: str
	doc_id: str
	score: float
	filepath: Optional[str] = None
	file_page: Optional[int] = None
	start_idx: Optional[int] = None
	end_idx: Optional[int] = None


class RunResponse(BaseModel):
	result: Union[str, List[str]]
	retrieved_passage: List[RetrievedPassage]


class RetrievalResponse(BaseModel):
	passages: List[RetrievedPassage]


class StreamResponse(BaseModel):
	"""
	When the type is generated_text, only generated_text is returned. The other fields are None.
	When the type is retrieved_passage, only retrieved_passage and passage_index are returned. The other fields are None.
	"""

	type: Literal["generated_text", "retrieved_passage"]
	generated_text: Optional[str]
	retrieved_passage: Optional[RetrievedPassage]
	passage_index: Optional[int]


class VersionResponse(BaseModel):
	version: str


class ApiRunner(BaseRunner):
	def __init__(self, config: Dict, project_dir: Optional[str] = None):
		super().__init__(config, project_dir)
		self.app = Quart(__name__)

		data_dir = os.path.join(project_dir, "data")
		self.corpus_df = pd.read_parquet(
			os.path.join(data_dir, "corpus.parquet"), engine="pyarrow"
		)
		self.__add_api_route()

	def __add_api_route(self):
		@self.app.route("/v1/run", methods=["POST"])
		async def run_query():
			try:
				data = await request.get_json()
				data = QueryRequest(**data)
			except ValidationError as e:
				return jsonify(e.errors()), 400

			previous_result = pd.DataFrame(
				{
					"qid": str(uuid.uuid4()),
					"query": [data.query],
					"retrieval_gt": [[]],
					"generation_gt": [""],
				}
			)  # pseudo qa data for execution
			for module_instance, module_param in zip(
				self.module_instances, self.module_params
			):
				new_result = module_instance.pure(
					previous_result=previous_result, **module_param
				)
				duplicated_columns = previous_result.columns.intersection(
					new_result.columns
				)
				drop_previous_result = previous_result.drop(columns=duplicated_columns)
				previous_result = pd.concat([drop_previous_result, new_result], axis=1)

			# Simulate processing the query
			generated_text = previous_result[data.result_column].tolist()[0]
			retrieved_passage = self.extract_retrieve_passage(previous_result)

			response = RunResponse(
				result=generated_text, retrieved_passage=retrieved_passage
			)

			return jsonify(response.model_dump()), 200

		@self.app.route("/v1/retrieve", methods=["POST"])
		async def run_retrieve_only():
			data = await request.get_json()
			query = data.get("query", None)
			if query is None:
				return jsonify(
					{
						"error": "Invalid request. You need to include 'query' in the request body."
					}
				), 400

			previous_result = pd.DataFrame(
				{
					"qid": str(uuid.uuid4()),
					"query": [query],
					"retrieval_gt": [[]],
					"generation_gt": [""],
				}
			)  # pseudo qa data for execution
			for module_instance, module_param in zip(
				self.module_instances, self.module_params
			):
				if isinstance(module_instance, BasePromptMaker) or isinstance(
					module_instance, BaseGenerator
				):
					continue
				new_result = module_instance.pure(
					previous_result=previous_result, **module_param
				)
				duplicated_columns = previous_result.columns.intersection(
					new_result.columns
				)
				drop_previous_result = previous_result.drop(columns=duplicated_columns)
				previous_result = pd.concat([drop_previous_result, new_result], axis=1)

			# Simulate processing the query
			retrieved_passages = self.extract_retrieve_passage(previous_result)

			retrieval_response = RetrievalResponse(passages=retrieved_passages)
			return jsonify(retrieval_response.model_dump()), 200

		@self.app.route("/v1/stream", methods=["POST"])
		async def stream_query():
			try:
				data = await request.get_json()
				data = QueryRequest(**data)
			except ValidationError as e:
				return jsonify(e.errors()), 400

			@stream_with_context
			async def generate():
				previous_result = pd.DataFrame(
					{
						"qid": str(uuid.uuid4()),
						"query": [data.query],
						"retrieval_gt": [[]],
						"generation_gt": [""],
					}
				)  # pseudo qa data for execution

				for module_instance, module_param in zip(
					self.module_instances, self.module_params
				):
					if not isinstance(module_instance, BaseGenerator):
						new_result = module_instance.pure(
							previous_result=previous_result, **module_param
						)
						duplicated_columns = previous_result.columns.intersection(
							new_result.columns
						)
						drop_previous_result = previous_result.drop(
							columns=duplicated_columns
						)
						previous_result = pd.concat(
							[drop_previous_result, new_result], axis=1
						)
					else:
						retrieved_passages = self.extract_retrieve_passage(
							previous_result
						)
						for i, retrieved_passage in enumerate(retrieved_passages):
							yield (
								StreamResponse(
									type="retrieved_passage",
									generated_text=None,
									retrieved_passage=retrieved_passage,
									passage_index=i,
								)
								.model_dump_json()
								.encode("utf-8")
							)
						# Start streaming of the result
						assert len(previous_result) == 1
						prompt: str = previous_result["prompts"].tolist()[0]
						async for delta in module_instance.astream(
							prompt=prompt, **module_param
						):
							response = StreamResponse(
								type="generated_text",
								generated_text=delta,
								retrieved_passage=None,
								passage_index=None,
							)
							yield response.model_dump_json().encode("utf-8")

			return generate(), 200, {"X-Something": "value"}

		@self.app.route("/version", methods=["GET"])
		def get_version():
			with open(VERSION_PATH, "r") as f:
				version = f.read().strip()
			response = VersionResponse(version=version)
			return jsonify(response.model_dump()), 200

	def run_api_server(
		self, host: str = "0.0.0.0", port: int = 8000, remote: bool = True, **kwargs
	):
		"""
		Run the pipeline as an api server.
		Here is api endpoint documentation => https://marker-inc-korea.github.io/AutoRAG/deploy/api_endpoint.html

		:param host: The host of the api server.
		:param port: The port of the api server.
		:param remote: Whether to expose the api server to the public internet using ngrok.
		:param kwargs: Other arguments for Flask app.run.
		"""
		logger.info(f"Run api server at {host}:{port}")
		if remote:
			from pyngrok import ngrok

			http_tunnel = ngrok.connect(str(port), "http")
			public_url = http_tunnel.public_url
			logger.info(f"Public API URL: {public_url}")
		self.app.run(host=host, port=port, **kwargs)

	def extract_retrieve_passage(self, df: pd.DataFrame) -> List[RetrievedPassage]:
		retrieved_ids: List[str] = df["retrieved_ids"].tolist()[0]
		contents = fetch_contents(self.corpus_df, [retrieved_ids])[0]
		scores = df["retrieve_scores"].tolist()[0]
		if "path" in self.corpus_df.columns:
			paths = fetch_contents(self.corpus_df, [retrieved_ids], column_name="path")[
				0
			]
		else:
			paths = [None] * len(retrieved_ids)
		metadatas = fetch_contents(
			self.corpus_df, [retrieved_ids], column_name="metadata"
		)[0]
		if "start_end_idx" in self.corpus_df.columns:
			start_end_indices = fetch_contents(
				self.corpus_df, [retrieved_ids], column_name="start_end_idx"
			)[0]
		else:
			start_end_indices = [None] * len(retrieved_ids)
		start_end_indices = to_list(start_end_indices)
		return list(
			map(
				lambda content,
				doc_id,
				score,
				path,
				metadata,
				start_end_idx: RetrievedPassage(
					content=content,
					doc_id=doc_id,
					score=score,
					filepath=path,
					file_page=metadata.get("page", None),
					start_idx=start_end_idx[0] if start_end_idx else None,
					end_idx=start_end_idx[1] if start_end_idx else None,
				),
				contents,
				retrieved_ids,
				scores,
				paths,
				metadatas,
				start_end_indices,
			)
		)
