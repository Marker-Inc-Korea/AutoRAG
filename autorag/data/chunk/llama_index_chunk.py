import os.path
from itertools import accumulate, chain
from typing import Tuple, List, Dict, Any, Optional

from llama_index.core import Document
from llama_index.core.node_parser.interface import NodeParser

from autorag.utils.util import process_batch, get_event_loop
from autorag.data.chunk.base import chunker_node, add_file_name
from autorag.data.utils.util import add_essential_metadata_llama_text_node


@chunker_node
def llama_index_chunk(
	texts: List[str],
	chunker: NodeParser,
	file_name_language: Optional[str] = None,
	metadata_list: Optional[List[Dict[str, str]]] = None,
	batch: int = 8,
) -> Tuple[
	List[str], List[str], List[str], List[Tuple[int, int]], List[Dict[str, Any]]
]:
	"""
	Chunk texts from the parsed result to use llama index chunk method

	:param texts: The list of texts to chunk from the parsed result
	:param chunker: A llama index NodeParser(Chunker) instance.
	:param file_name_language: The language to use 'add_file_name' feature.
	    You need to set one of 'English' and 'Korean'
	    The 'add_file_name' feature is to add a file_name to chunked_contents.
	    This is used to prevent hallucination by retrieving contents from the wrong document.
	    Default form of 'English' is "file_name: {file_name}\n contents: {content}"
	:param metadata_list: The list of dict of metadata from the parsed result
	:param batch: The batch size for chunk texts. Default is 8
	:return: tuple of lists containing the chunked doc_id, contents, path, start_idx, end_idx and metadata
	"""
	tasks = [
		llama_index_chunk_pure(text, chunker, file_name_language, meta)
		for text, meta in zip(texts, metadata_list)
	]
	loop = get_event_loop()
	results = loop.run_until_complete(process_batch(tasks, batch))

	doc_id, contents, path, start_end_idx, metadata = (
		list(chain.from_iterable(item)) for item in zip(*results)
	)

	return list(doc_id), list(contents), list(path), list(start_end_idx), list(metadata)


async def llama_index_chunk_pure(
	text: str,
	chunker: NodeParser,
	file_name_language: Optional[str] = None,
	_metadata: Optional[Dict[str, str]] = None,
):
	# set document
	document = [Document(text=text, metadata=_metadata)]

	# chunk document
	chunk_results = chunker.get_nodes_from_documents(documents=document)

	# make doc_id
	doc_id = list(map(lambda node: node.node_id, chunk_results))

	# make path
	path_lst = list(map(lambda x: x.metadata.get("path", ""), chunk_results))

	# make contents and start_end_idx
	if file_name_language:
		chunked_file_names = list(map(lambda x: os.path.basename(x), path_lst))
		chunked_texts = list(map(lambda x: x.text, chunk_results))
		start_end_idx = gen_start_end_idx(chunked_texts)
		contents = add_file_name(file_name_language, chunked_file_names, chunked_texts)
	else:
		contents = list(map(lambda x: x.text, chunk_results))
		start_end_idx = gen_start_end_idx(contents)

	metadata = list(
		map(
			lambda node: add_essential_metadata_llama_text_node(
				node.metadata, node.relationships
			),
			chunk_results,
		)
	)

	return doc_id, contents, path_lst, start_end_idx, metadata


def gen_start_end_idx(contents: List[str]) -> List[Tuple[int, int]]:
	lengths = [len(content) for content in contents]
	start_indices = list(accumulate([0] + lengths[:-1]))
	return [
		(start, start + length - 1) for start, length in zip(start_indices, lengths)
	]
