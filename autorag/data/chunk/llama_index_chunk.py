import os.path
from typing import Tuple, List, Dict, Any, Optional

from llama_index.core import Document
from llama_index.core.node_parser.interface import NodeParser

from autorag.data.chunk.base import chunker_node, add_file_name
from autorag.data.utils.util import add_essential_metadata_llama_text_node


@chunker_node
def llama_index_chunk(
	texts: List[str],
	chunker: NodeParser,
	file_name_language: Optional[str] = None,
	metadata_list: Optional[List[Dict[str, str]]] = None,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
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
	:return: tuple of lists containing the chunked doc_id, contents, and metadata
	"""
	# set documents
	documents = [
		Document(text=text, metadata=metadata)
		for text, metadata in zip(texts, metadata_list)
	]

	# chunk documents
	chunk_results = chunker.get_nodes_from_documents(documents=documents)

	# make doc_id
	doc_id = list(map(lambda node: node.node_id, chunk_results))

	# make contents
	if file_name_language:
		path_lst = list(map(lambda x: x.metadata.get("path", ""), chunk_results))
		chunked_file_names = list(map(lambda x: os.path.basename(x), path_lst))
		chunked_texts = list(map(lambda x: x.text, chunk_results))
		contents = add_file_name(file_name_language, chunked_file_names, chunked_texts)
	else:
		contents = list(map(lambda node: node.text, chunk_results))

	# make metadata
	metadata = list(
		map(
			lambda node: add_essential_metadata_llama_text_node(
				node.metadata, node.relationships
			),
			chunk_results,
		)
	)
	return doc_id, contents, metadata
