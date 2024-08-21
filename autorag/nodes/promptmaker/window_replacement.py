import logging
from typing import List, Dict

from autorag.nodes.promptmaker.base import prompt_maker_node

logger = logging.getLogger("AutoRAG")


@prompt_maker_node
def window_replacement(
	prompt: str,
	queries: List[str],
	retrieved_contents: List[List[str]],
	retrieved_metadata: List[List[Dict]],
) -> List[str]:
	"""
	Replace retrieved_contents with window to create a Prompt
	(only available for corpus chunked with Sentence window method)
	You must type a prompt or prompt list at config yaml file like this:

	.. Code:: yaml
	nodes:
	- node_type: prompt_maker
	  modules:
	  - module_type: window_replacement
	    prompt: [Answer this question: {query} \n\n {retrieved_contents},
	    Read the passages carefully and answer this question: {query} \n\n Passages: {retrieved_contents}]

	:param prompt: A prompt string.
	:param queries: List of query strings.
	:param retrieved_contents: List of retrieved contents.
	:param retrieved_metadata: List of retrieved metadata.
	:return: Prompts that made by window_replacement.
	"""

	def window_replacement_row(
		_prompt: str, _query: str, _retrieved_contents, _retrieved_metadata: List[Dict]
	) -> str:
		window_list = []
		for content, metadata in zip(_retrieved_contents, _retrieved_metadata):
			if "window" in metadata:
				window_list.append(metadata["window"])
			else:
				window_list.append(content)
				logger.info(
					"Only available for corpus chunked with Sentence window method."
					"window_replacement will not proceed."
				)
		contents_str = "\n\n".join(window_list)
		return _prompt.format(query=_query, retrieved_contents=contents_str)

	return list(
		map(
			lambda x: window_replacement_row(prompt, x[0], x[1], x[2]),
			zip(queries, retrieved_contents, retrieved_metadata),
		)
	)
