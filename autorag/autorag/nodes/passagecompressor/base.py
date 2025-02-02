import abc
import logging
from typing import Dict

import pandas as pd
from llama_index.core.llms import LLM

from autorag import generator_models
from autorag.schema import BaseModule
from autorag.utils import result_to_dataframe

logger = logging.getLogger("AutoRAG")


class BasePassageCompressor(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: str, *args, **kwargs):
		logger.info(
			f"Initialize passage compressor node - {self.__class__.__name__} module..."
		)

	def __del__(self):
		logger.info(
			f"Deleting passage compressor node - {self.__class__.__name__} module..."
		)

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		logger.info(
			f"Running passage compressor node - {self.__class__.__name__} module..."
		)
		assert all(
			[
				column in previous_result.columns
				for column in [
					"query",
					"retrieved_contents",
				]
			]
		), "previous_result must have retrieved_contents, retrieved_ids, and retrieve_scores columns."
		assert len(previous_result) > 0, "previous_result must have at least one row."

		queries = previous_result["query"].tolist()
		retrieved_contents = previous_result["retrieved_contents"].tolist()
		return queries, retrieved_contents


class LlamaIndexCompressor(BasePassageCompressor, metaclass=abc.ABCMeta):
	param_list = ["prompt", "chat_prompt", "batch"]

	def __init__(self, project_dir: str, **kwargs):
		"""
		Initialize passage compressor module.

		:param project_dir: The project directory
		:param llm: The llm name that will be used to summarize.
			The LlamaIndex LLM model can be used in here.
		:param kwargs: Extra parameter for init llm
		"""
		super().__init__(project_dir)
		kwargs_dict = dict(
			filter(lambda x: x[0] not in self.param_list, kwargs.items())
		)
		llm_name = kwargs_dict.pop("llm")
		self.llm: LLM = make_llm(llm_name, kwargs_dict)

	def __del__(self):
		del self.llm
		super().__del__()

	@result_to_dataframe(["retrieved_contents"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries, retrieved_contents = self.cast_to_run(previous_result)
		param_dict = dict(filter(lambda x: x[0] in self.param_list, kwargs.items()))
		result = self._pure(queries, retrieved_contents, **param_dict)
		return list(map(lambda x: [x], result))


def make_llm(llm_name: str, kwargs: Dict) -> LLM:
	if llm_name not in generator_models:
		raise KeyError(
			f"{llm_name} is not supported. "
			"You can add it manually by calling autorag.generator_models."
		)
	return generator_models[llm_name](**kwargs)
