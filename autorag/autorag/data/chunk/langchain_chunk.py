import os
from itertools import chain
import uuid
from typing import Tuple, List, Dict, Any, Optional

from langchain_text_splitters import TextSplitter

from autorag.data.chunk.base import chunker_node, add_file_name
from autorag.data.utils.util import add_essential_metadata, get_start_end_idx


@chunker_node
def langchain_chunk(
	texts: List[str],
	chunker: TextSplitter,
	file_name_language: Optional[str] = None,
	metadata_list: Optional[List[Dict[str, str]]] = None,
) -> Tuple[
	List[str], List[str], List[str], List[Tuple[int, int]], List[Dict[str, Any]]
]:
	"""
	Chunk texts from the parsed result to use langchain chunk method

	:param texts: The list of texts to chunk from the parsed result
	:param chunker: A langchain TextSplitter(Chunker) instance.
	:param file_name_language: The language to use 'add_file_name' feature.
	    You need to set one of 'English' and 'Korean'
	    The 'add_file_name' feature is to add a file_name to chunked_contents.
	    This is used to prevent hallucination by retrieving contents from the wrong document.
	    Default form of 'English' is "file_name: {file_name}\n contents: {content}"
	:param metadata_list: The list of dict of metadata from the parsed result
	:return: tuple of lists containing the chunked doc_id, contents, path, start_idx, end_idx and metadata
	"""
	results = [
		langchain_chunk_pure(text, chunker, file_name_language, meta)
		for text, meta in zip(texts, metadata_list)
	]

	doc_id, contents, path, start_end_idx, metadata = (
		list(chain.from_iterable(item)) for item in zip(*results)
	)

	return doc_id, contents, path, start_end_idx, metadata


def langchain_chunk_pure(
	text: str,
	chunker: TextSplitter,
	file_name_language: Optional[str] = None,
	_metadata: Optional[Dict[str, str]] = None,
):
	# chunk
	chunk_results = chunker.create_documents([text], metadatas=[_metadata])

	# make doc_id
	doc_id = list(str(uuid.uuid4()) for _ in range(len(chunk_results)))

	# make path
	path_lst = list(map(lambda x: x.metadata.get("path", ""), chunk_results))

	# make contents and start_end_idx
	if file_name_language:
		chunked_file_names = list(map(lambda x: os.path.basename(x), path_lst))
		chunked_texts = list(map(lambda x: x.page_content, chunk_results))
		start_end_idx = list(map(lambda x: get_start_end_idx(text, x), chunked_texts))
		contents = add_file_name(file_name_language, chunked_file_names, chunked_texts)
	else:
		contents = list(map(lambda node: node.page_content, chunk_results))
		start_end_idx = list(map(lambda x: get_start_end_idx(text, x), contents))

	# make metadata
	metadata = list(
		map(lambda node: add_essential_metadata(node.metadata), chunk_results)
	)

	return doc_id, contents, path_lst, start_end_idx, metadata
