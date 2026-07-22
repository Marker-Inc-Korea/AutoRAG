import os
from typing import List, Tuple
from itertools import chain

from llama_parse import LlamaParse

from autorag.data.parse.base import parser_node
from autorag.utils.util import process_batch, get_event_loop


@parser_node
def llama_parse(
	data_path_list: List[str],
	batch: int = 8,
	use_vendor_multimodal_model: bool = False,
	vendor_multimodal_model_name: str = "openai-gpt4o",
	use_own_key: bool = False,
	vendor_multimodal_api_key: str = None,
	**kwargs,
) -> Tuple[List[str], List[str], List[int]]:
	"""
	Parse documents to use llama_parse.
	LLAMA_CLOUD_API_KEY environment variable should be set.
	You can get the key from https://cloud.llamaindex.ai/api-key

	:param data_path_list: The list of data paths to parse.
	:param batch: The batch size for parse documents. Default is 8.
	:param use_vendor_multimodal_model: Whether to use the vendor multimodal model. Default is False.
	:param vendor_multimodal_model_name: The name of the vendor multimodal model. Default is "openai-gpt4o".
	:param use_own_key: Whether to use the own API key. Default is False.
	:param vendor_multimodal_api_key: The API key for the vendor multimodal model.
	:param kwargs: The extra parameters for creating the llama_parse instance.
	:return: tuple of lists containing the parsed texts, path and pages.
	"""
	if use_vendor_multimodal_model:
		kwargs = _add_multimodal_params(
			kwargs,
			use_vendor_multimodal_model,
			vendor_multimodal_model_name,
			use_own_key,
			vendor_multimodal_api_key,
		)

	parse_instance = LlamaParse(**kwargs)

	tasks = [
		llama_parse_pure(data_path, parse_instance) for data_path in data_path_list
	]
	loop = get_event_loop()
	results = loop.run_until_complete(process_batch(tasks, batch))

	del parse_instance

	texts, path, pages = (list(chain.from_iterable(item)) for item in zip(*results))

	return texts, path, pages


async def llama_parse_pure(
	data_path: str, parse_instance
) -> Tuple[List[str], List[str], List[int]]:
	documents = await parse_instance.aload_data(data_path)

	texts = list(map(lambda x: x.text, documents))
	path = [data_path] * len(texts)
	pages = list(range(1, len(documents) + 1))

	return texts, path, pages


def _add_multimodal_params(
	kwargs,
	use_vendor_multimodal_model,
	vendor_multimodal_model_name,
	use_own_key,
	vendor_multimodal_api_key,
) -> dict:
	kwargs["use_vendor_multimodal_model"] = use_vendor_multimodal_model
	kwargs["vendor_multimodal_model_name"] = vendor_multimodal_model_name

	def set_multimodal_api_key(
		multimodal_model_name: str = "openai-gpt4o", _api_key: str = None
	) -> str:
		if multimodal_model_name in ["openai-gpt4o", "openai-gpt-4o-mini"]:
			_api_key = (
				os.getenv("OPENAI_API_KEY", None) if _api_key is None else _api_key
			)
			if _api_key is None:
				raise KeyError(
					"Please set the OPENAI_API_KEY in the environment variable OPENAI_API_KEY "
					"or directly set it on the config YAML file."
				)
		elif multimodal_model_name in ["anthropic-sonnet-3.5"]:
			_api_key = (
				os.getenv("ANTHROPIC_API_KEY", None) if _api_key is None else _api_key
			)
			if _api_key is None:
				raise KeyError(
					"Please set the ANTHROPIC_API_KEY in the environment variable ANTHROPIC_API_KEY "
					"or directly set it on the config YAML file."
				)
		elif multimodal_model_name in ["gemini-1.5-flash", "gemini-1.5-pro"]:
			_api_key = (
				os.getenv("GEMINI_API_KEY", None) if _api_key is None else _api_key
			)
			if _api_key is None:
				raise KeyError(
					"Please set the GEMINI_API_KEY in the environment variable GEMINI_API_KEY "
					"or directly set it on the config YAML file."
				)
		elif multimodal_model_name in ["custom-azure-model"]:
			raise NotImplementedError(
				"Custom Azure multimodal model is not supported yet."
			)
		else:
			raise ValueError("Invalid multimodal model name.")

		return _api_key

	if use_own_key:
		api_key = set_multimodal_api_key(
			vendor_multimodal_model_name, vendor_multimodal_api_key
		)
		kwargs["vendor_multimodal_api_key"] = api_key

	return kwargs
