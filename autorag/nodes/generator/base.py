import abc
import functools
import logging
from pathlib import Path
from typing import Union, Tuple, List

import pandas as pd
from llama_index.core.output_parsers import PydanticOutputParser

from autorag import generator_models
from autorag.schema import BaseModule
from autorag.utils import result_to_dataframe

logger = logging.getLogger("AutoRAG")


class BaseGenerator(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: str, llm: str, *args, **kwargs):
		logger.info(f"Initialize generator node - {self.__class__.__name__}")
		self.llm = llm

	def __del__(self):
		logger.info(f"Deleting generator module - {self.__class__.__name__}")

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		logger.info(f"Running generator node - {self.__class__.__name__} module...")
		assert (
			"prompts" in previous_result.columns
		), "previous_result must contain prompts column."
		prompts = previous_result["prompts"].tolist()
		return prompts

	def structured_output(self, prompts: List[str], output_cls):
		response, _, _ = self._pure(prompts)
		parser = PydanticOutputParser(output_cls)
		result = []
		for res in response:
			try:
				result.append(parser.parse(res))
			except Exception as e:
				logger.warning(
					f"Error parsing response: {e} \nSo returning None instead in this case."
				)
				result.append(None)
		return result

	@abc.abstractmethod
	async def astream(self, prompt: str, **kwargs):
		pass

	@abc.abstractmethod
	def stream(self, prompt: str, **kwargs):
		pass


def generator_node(func):
	@functools.wraps(func)
	@result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
	def wrapper(
		project_dir: Union[str, Path], previous_result: pd.DataFrame, llm: str, **kwargs
	) -> Tuple[List[str], List[List[int]], List[List[float]]]:
		"""
		This decorator makes a generator module to be a node.
		It automatically extracts prompts from previous_result and runs the generator function.
		Plus, it retrieves the llm instance from autorag.generator_models.

		:param project_dir: The project directory.
		:param previous_result: The previous result that contains prompts,
		:param llm: The llm name that you want to use.
		:param kwargs: The extra parameters for initializing the llm instance.
		:return: Pandas dataframe that contains generated texts, generated tokens, and generated log probs.
		    Each column is "generated_texts", "generated_tokens", and "generated_log_probs".
		"""
		logger.info(f"Running generator node - {func.__name__} module...")
		assert (
			"prompts" in previous_result.columns
		), "previous_result must contain prompts column."
		prompts = previous_result["prompts"].tolist()
		if func.__name__ == "llama_index_llm":
			if llm not in generator_models:
				raise ValueError(
					f"{llm} is not a valid llm name. Please check the llm name."
					"You can check valid llm names from autorag.generator_models."
				)
			batch = kwargs.pop("batch", 16)
			if llm == "huggingfacellm":
				model_name = kwargs.pop("model", None)
				if model_name is not None:
					kwargs["model_name"] = model_name
				else:
					if "model_name" not in kwargs.keys():
						raise ValueError(
							"`model` or `model_name` parameter must be provided for using huggingfacellm."
						)
				kwargs["tokenizer_name"] = kwargs["model_name"]
			llm_instance = generator_models[llm](**kwargs)
			result = func(prompts=prompts, llm=llm_instance, batch=batch)
			del llm_instance
			return result
		else:
			return func(prompts=prompts, llm=llm, **kwargs)

	return wrapper
