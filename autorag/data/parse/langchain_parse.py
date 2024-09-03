import multiprocessing as mp
import os
from typing import List, Tuple

from autorag.data import parse_modules
from autorag.data.parse.base import parser_node


@parser_node
def langchain_parse(
	data_path_list: List[str], parse_method: str, **kwargs
) -> Tuple[List[str], List[str]]:
	if parse_method in ["directory", "unstructured"]:
		results = parse_all_files(data_path_list, parse_method, **kwargs)
		texts, file_names = results[0], results[1]

	else:
		num_workers = mp.cpu_count()
		# Execute parallel processing
		with mp.Pool(num_workers) as pool:
			results = pool.starmap(
				langchain_parse_pure,
				[(data_path, parse_method, kwargs) for data_path in data_path_list],
			)

		texts, file_names = zip(*results)
		texts, file_names = list(texts), list(file_names)

	return texts, file_names


def langchain_parse_pure(data_path: str, parse_method: str, kwargs) -> Tuple[str, str]:
	"""
	Parses a single file using the specified parse method.

	Args:
	    data_path (str): The file path to parse.
	    parse_method (str): The parsing method to use.
	    kwargs (Dict): Additional keyword arguments for the parsing method.

	Returns:
	    Tuple[str, str]: A tuple containing the parsed text and the file path.
	"""

	parse_instance = parse_modules[parse_method](data_path, **kwargs)

	# Load the text from the file
	documents = parse_instance.load()
	text = documents[0].page_content
	file_name = os.path.basename(data_path)

	# Clean up the parse instance
	del parse_instance

	return text, file_name


def parse_all_files(
	data_path_list: List[str], parse_method: str, **kwargs
) -> Tuple[List[str], List[str]]:
	if parse_method == "unstructured":
		parse_instance = parse_modules[parse_method](data_path_list, **kwargs)
	elif parse_method == "directory":
		parse_instance = parse_modules[parse_method](**kwargs)
	else:
		raise ValueError(f"Unsupported parse method: {parse_method}")
	docs = parse_instance.load()
	texts = [doc.page_content for doc in docs]
	file_names = [doc.metadata["source"].split("/")[-1] for doc in docs]

	del parse_instance
	return texts, file_names
