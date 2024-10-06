import logging
import uuid

import pandas as pd

from autorag.deploy.base import BaseRunner

import gradio as gr


logger = logging.getLogger("AutoRAG")


class GradioRunner(BaseRunner):
	def run_web(
		self,
		server_name: str = "0.0.0.0",
		server_port: int = 7680,
		share: bool = False,
		**kwargs,
	):
		"""
		Run web interface to interact pipeline.
		You can access the web interface at `http://server_name:server_port` in your browser

		:param server_name: The host of the web. Default is 0.0.0.0.
		:param server_port: The port of the web. Default is 7680.
		:param share: Whether to create a publicly shareable link. Default is False.
		:param kwargs: Other arguments for gr.ChatInterface.launch.
		"""

		logger.info(f"Run web interface at http://{server_name}:{server_port}")

		def get_response(message, _):
			return self.run(message)

		gr.ChatInterface(
			get_response, title="📚 AutoRAG", retry_btn=None, undo_btn=None
		).launch(
			server_name=server_name, server_port=server_port, share=share, **kwargs
		)

	def run(self, query: str, result_column: str = "generated_texts"):
		"""
		Run the pipeline with query.
		The loaded pipeline must start with a single query,
		so the first module of the pipeline must be `query_expansion` or `retrieval` module.

		:param query: The query of the user.
		:param result_column: The result column name for the answer.
		    Default is `generated_texts`, which is the output of the `generation` module.
		:return: The result of the pipeline.
		"""
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
			new_result = module_instance.pure(
				previous_result=previous_result, **module_param
			)
			duplicated_columns = previous_result.columns.intersection(
				new_result.columns
			)
			drop_previous_result = previous_result.drop(columns=duplicated_columns)
			previous_result = pd.concat([drop_previous_result, new_result], axis=1)

		return previous_result[result_column].tolist()[0]
