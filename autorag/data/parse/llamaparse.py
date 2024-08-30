import asyncio
from typing import List, Tuple

import nest_asyncio
from llama_parse import LlamaParse

from autorag.data.parse.base import parser_node
from autorag.utils.util import process_batch

nest_asyncio.apply()


@parser_node
def llama_parse(
	data_path_list: List[str], batch: int = 8, **kwargs
) -> Tuple[List[str], List[str]]:
	parse_instance = LlamaParse(**kwargs)

	tasks = [
		llama_parse_pure(data_path, parse_instance) for data_path in data_path_list
	]
	loop = asyncio.get_event_loop()
	results = loop.run_until_complete(process_batch(tasks, batch))

	del parse_instance

	texts, names = zip(*results)

	return list(texts), list(names)


async def llama_parse_pure(data_path: str, parse_instance) -> Tuple[str, str]:
	documents = await parse_instance.aload_data(data_path)

	text = documents[0].text
	file_name = data_path.split("/")[-1]

	return text, file_name
