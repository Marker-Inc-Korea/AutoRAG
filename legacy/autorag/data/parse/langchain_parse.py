import multiprocessing as mp
from itertools import chain
from typing import List, Tuple

from autorag.data import parse_modules
from autorag.data.parse.base import parser_node


@parser_node
def langchain_parse(
	data_path_list: List[str], parse_method: str, **kwargs
) -> Tuple[List[str], List[str], List[int]]:
	"""
	Parse documents to use langchain document_loaders(parse) method

	:param data_path_list: The list of data paths to parse.
	:param parse_method: A langchain document_loaders(parse) method to use.
	:param kwargs: The extra parameters for creating the langchain document_loaders(parse) instance.
	:return: tuple of lists containing the parsed texts, path and pages.
	"""
	if parse_method in ["directory", "unstructured"]:
		results = parse_all_files(data_path_list, parse_method, **kwargs)
		texts, path = results[0], results[1]
		pages = [-1] * len(texts)

	else:
		num_workers = mp.cpu_count()
		# Execute parallel processing
		with mp.Pool(num_workers) as pool:
			results = pool.starmap(
				langchain_parse_pure,
				[(data_path, parse_method, kwargs) for data_path in data_path_list],
			)

		texts, path, pages = (list(chain.from_iterable(item)) for item in zip(*results))

	return texts, path, pages


def langchain_parse_pure(
	data_path: str, parse_method: str, kwargs
) -> Tuple[List[str], List[str], List[int]]:
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

	texts = list(map(lambda x: x.page_content, documents))
	path = [data_path] * len(texts)
	if parse_method in ["pymupdf", "pdfplumber", "pypdf", "pypdfium2"]:
		pages = list(range(1, len(documents) + 1))
	else:
		pages = [-1] * len(texts)

	# Clean up the parse instance
	del parse_instance

	return texts, path, pages


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
	file_names = [doc.metadata["source"] for doc in docs]

	del parse_instance
	return texts, file_names
