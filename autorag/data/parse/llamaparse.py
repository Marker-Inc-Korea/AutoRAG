from typing import List, Tuple

from llama_parse import LlamaParse

from autorag.data.parse.base import parser_node
from autorag.utils.util import process_batch, get_event_loop


@parser_node
def llama_parse(
	data_path_list: List[str], batch: int = 8, **kwargs
) -> Tuple[List[str], List[str], List[int]]:
	"""
	Parse documents to use llama_parse.
	LLAMA_CLOUD_API_KEY environment variable should be set.
	You can get the key from https://cloud.llamaindex.ai/api-key

	:param data_path_list: The list of data paths to parse.
	:param batch: The batch size for parse documents. Default is 8.
	:param kwargs: The extra parameters for creating the llama_parse instance.
	:return: tuple of lists containing the parsed texts, path and pages.
	"""
	parse_instance = LlamaParse(**kwargs)

	tasks = [
		llama_parse_pure(data_path, parse_instance) for data_path in data_path_list
	]
	loop = get_event_loop()
	results = loop.run_until_complete(process_batch(tasks, batch))

	del parse_instance

	texts, path = zip(*results)
	pages = [-1] * len(texts)

	return list(texts), list(path), pages


async def llama_parse_pure(data_path: str, parse_instance) -> Tuple[str, str]:
	documents = await parse_instance.aload_data(data_path)

	text = documents[0].text

	return text, data_path
