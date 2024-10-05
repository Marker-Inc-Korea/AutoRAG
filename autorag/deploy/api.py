import logging
import os
import pathlib
import uuid
from typing import Dict, Optional, List, Union

import pandas as pd
from flask_swagger_ui import get_swaggerui_blueprint
from flask import Flask, request, jsonify, Response, stream_with_context
from pydantic import BaseModel, ValidationError

from autorag.deploy.base import BaseRunner
from autorag.nodes.generator.base import BaseGenerator
from autorag.utils import fetch_contents
from autorag.utils.util import get_event_loop

logger = logging.getLogger("AutoRAG")

deploy_dir = pathlib.Path(__file__).parent
root_dir = pathlib.Path(__file__).parent.parent

SWAGGER_URL = "/api/docs"
API_URL = "/api/spec"
YAML_PATH = os.path.join(deploy_dir, "swagger.yaml")
VERSION_PATH = os.path.join(root_dir, "VERSION")


class QueryRequest(BaseModel):
	query: str
	result_column: Optional[str] = "generated_texts"


class RetrievedPassage(BaseModel):
	content: str
	doc_id: str
	filepath: Optional[str] = None
	file_page: Optional[int] = None
	start_idx: Optional[int] = None
	end_idx: Optional[int] = None


class RunResponse(BaseModel):
	result: Union[str, List[str]]
	retrieved_passage: List[RetrievedPassage]


class VersionResponse(BaseModel):
	version: str


empty_run_response = RunResponse(result="", retrieved_passage=[])


class ApiRunner(BaseRunner):
	def __init__(self, config: Dict, project_dir: Optional[str] = None):
		super().__init__(config, project_dir)
		self.app = Flask(__name__)

		data_dir = os.path.join(project_dir, "data")
		self.corpus_df = pd.read_parquet(
			os.path.join(data_dir, "corpus.parquet"), engine="pyarrow"
		)

		with open(VERSION_PATH, "r") as f:
			version = f.read().strip()

		swagger_ui_blueprint = get_swaggerui_blueprint(
			SWAGGER_URL,
			API_URL,
			config={
				"app_name": "AutoRAG API",
				"version": version,
			},
		)
		self.app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)
		self.__add_api_route()

	def __add_api_route(self):
		@self.app.route("/v1/run", methods=["POST"])
		def run_query():
			try:
				data = QueryRequest(**request.json)
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

		@self.app.route("/v1/stream", methods=["POST"])
		def stream_query():
			try:
				data = QueryRequest(**request.json)
			except ValidationError as e:
				return jsonify(e.errors()), 400

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
						response = RunResponse(
							result="", retrieved_passage=retrieved_passages
						)
						yield jsonify(response.model_dump()), 200
						# Start streaming of the result
						assert len(previous_result) == 1
						prompt: str = previous_result["prompts"].tolist()[0]
						stream = await module_instance.stream(
							prompt=prompt, **module_param
						)
						async for delta in stream:
							response = RunResponse(
								result=delta, retrieved_passage=empty_run_response
							)
							yield jsonify(response.model_dump()), 200

			loop = get_event_loop()
			iterator = iter_over_async(generate(), loop)
			ctx = stream_with_context(iterator)
			response = Response(ctx, content_type="text/plain")
			response.headers["X-Accel-Buffering"] = "no"
			response.headers["Transfer-Encoding"] = "chunked"
			return response

		@self.app.route("/version", methods=["GET"])
		def get_version():
			with open(VERSION_PATH, "r") as f:
				version = f.read().strip()
			response = VersionResponse(version=version)
			return jsonify(response.model_dump()), 200

	def run_api_server(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
		"""
		Run the pipeline as api server.
		You can send POST request to `http://host:port/run` with json body like below:

		.. Code:: json

		    {
		        "query": "your query",
		        "result_column": "generated_texts"
		    }

		And it returns json response like below:

		.. Code:: json

		    {
		        "answer": "your answer"
		    }

		:param host: The host of the api server.
		:param port: The port of the api server.
		:param kwargs: Other arguments for Flask app.run.
		"""
		logger.info(f"Run api server at {host}:{port}")
		self.app.run(host=host, port=port, **kwargs)

	def extract_retrieve_passage(self, df: pd.DataFrame) -> List[RetrievedPassage]:
		retrieved_ids: List[str] = df["retrieved_ids"].tolist()[0]
		contents = fetch_contents(self.corpus_df, [retrieved_ids])[0]
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
		return list(
			map(
				lambda content, doc_id, path, metadata, start_end_idx: RetrievedPassage(
					content=content,
					doc_id=doc_id,
					filepath=path,
					file_page=metadata.get("page", None),
					start_idx=start_end_idx[0] if start_end_idx else None,
					end_idx=start_end_idx[1] if start_end_idx else None,
				),
				contents,
				retrieved_ids,
				paths,
				metadatas,
				start_end_indices,
			)
		)


def iter_over_async(ait, loop):
	ait = ait.__aiter__()

	async def get_next():
		try:
			obj = await ait.__anext__()
			return False, obj
		except StopAsyncIteration:
			return True, None

	while True:
		done, obj = loop.run_until_complete(get_next())
		if done:
			break
		yield obj
