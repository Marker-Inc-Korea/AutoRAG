import functools
import logging
from pathlib import Path
from typing import List, Union, Dict, Optional

import pandas as pd

from autorag.support import get_support_modules
from autorag.utils import result_to_dataframe, validate_qa_dataset

logger = logging.getLogger("AutoRAG")


def query_expansion_node(func):
	@functools.wraps(func)
	@result_to_dataframe(["queries"])
	def wrapper(
		project_dir: Union[str, Path], previous_result: pd.DataFrame, *args, **kwargs
	) -> List[List[str]]:
		logger.info(f"Running query expansion node - {func.__name__} module...")
		validate_qa_dataset(previous_result)

		# find queries columns
		assert (
			"query" in previous_result.columns
		), "previous_result must have query column."
		queries = previous_result["query"].tolist()

		if func.__name__ == "pass_query_expansion":
			return func(queries=queries)

		# pop prompt from kwargs
		if "prompt" in kwargs.keys():
			prompt = kwargs.pop("prompt")
		else:
			prompt = ""

		# set generator module for query expansion
		generator_callable, generator_param = make_generator_callable_param(kwargs)

		# run query expansion function
		expanded_queries = func(
			queries=queries,
			prompt=prompt,
			generator_func=generator_callable,
			generator_params=generator_param,
		)
		# delete empty string in the nested expanded queries list
		expanded_queries = [
			list(map(lambda x: x.strip(), sublist)) for sublist in expanded_queries
		]
		expanded_queries = [
			list(filter(lambda x: bool(x), sublist)) for sublist in expanded_queries
		]
		return expanded_queries

	return wrapper


def make_generator_callable_param(generator_dict: Optional[Dict]):
	if "generator_module_type" not in generator_dict.keys():
		generator_dict = {
			"generator_module_type": "llama_index_llm",
			"llm": "openai",
			"model": "gpt-3.5-turbo",
		}
	module_str = generator_dict.pop("generator_module_type")
	module_callable = get_support_modules(module_str)
	module_param = generator_dict
	return module_callable, module_param
