import os
import uuid
from typing import Tuple, List, Dict, Any, Optional

from langchain_text_splitters import TextSplitter

from autorag.data.chunk.base import chunker_node, add_file_name
from autorag.data.utils.util import add_essential_metadata


@chunker_node
def langchain_chunk(
	texts: List[str],
	chunker: TextSplitter,
	file_name_language: Optional[str] = None,
	metadata_list: Optional[List[Dict[str, str]]] = None,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
	chunk_results = chunker.create_documents(texts, metadatas=metadata_list)

	# make doc_id
	doc_id = list(str(uuid.uuid4()) for _ in range(len(chunk_results)))

	if file_name_language:
		path_lst = list(map(lambda x: x.metadata.get("path", ""), chunk_results))
		chunked_file_names = list(map(lambda x: os.path.basename(x), path_lst))
		chunked_texts = list(map(lambda x: x.page_content, chunk_results))
		contents = add_file_name(file_name_language, chunked_file_names, chunked_texts)
	else:
		contents = list(map(lambda node: node.page_content, chunk_results))

	# make metadata
	metadata = list(
		map(lambda node: add_essential_metadata(node.metadata), chunk_results)
	)

	return doc_id, contents, metadata
