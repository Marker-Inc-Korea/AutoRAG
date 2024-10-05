import logging
import os
import pathlib
from typing import Dict, Optional

from flask import Flask, request
from flask_swagger_ui import get_swaggerui_blueprint
from pydantic import BaseModel

from autorag.deploy.base import BaseRunner

logger = logging.getLogger("AutoRAG")

deploy_dir = pathlib.Path(__file__).parent

SWAGGER_URL = "/api/docs"
API_URL = "/api/spec"
YAML_PATH = os.path.join(deploy_dir, "swagger.yaml")


class ApiRunner(BaseRunner):
	def __init__(self, config: Dict, project_dir: Optional[str] = None):
		super().__init__(config, project_dir)
		self.app = Flask(__name__)

		swagger_ui_blueprint = get_swaggerui_blueprint(
			SWAGGER_URL,
			API_URL,
			config={
				"app_name": "AutoRAG API",
				"version": 0.3,
			},
		)
		self.app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)
		self.__add_api_route()

	def __add_api_route(self):
		@self.app.route("/run", methods=["POST"])
		def run_pipeline():
			runner_input = RunnerInput(**request.json)
			query = runner_input.query
			result_column = runner_input.result_column
			result = self.run(query, result_column)
			return {result_column: result}

		# TODO: Separate between retrieval and generation

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


class RunnerInput(BaseModel):
	query: str
	result_column: str = "generated_texts"
