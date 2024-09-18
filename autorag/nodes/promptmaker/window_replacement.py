import logging
import os
from typing import List, Dict

import pandas as pd

from autorag.nodes.promptmaker.base import BasePromptMaker
from autorag.utils import result_to_dataframe, fetch_contents

logger = logging.getLogger("AutoRAG")


class WindowReplacement(BasePromptMaker):
	def __init__(self, project_dir: str, *args, **kwargs):
		super().__init__(project_dir, *args, **kwargs)
		# load corpus
		data_dir = os.path.join(project_dir, "data")
		self.corpus_data = pd.read_parquet(
			os.path.join(data_dir, "corpus.parquet"), engine="pyarrow"
		)

	@result_to_dataframe(["prompts"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		query, retrieved_contents, prompt = self.cast_to_run(
			previous_result, *args, **kwargs
		)
		retrieved_ids = previous_result["retrieved_ids"].tolist()
		# get metadata from corpus
		retrieved_metadata = fetch_contents(
			self.corpus_data, retrieved_ids, column_name="metadata"
		)
		return self._pure(prompt, query, retrieved_contents, retrieved_metadata)

	def _pure(
		self,
		prompt: str,
		queries: List[str],
		retrieved_contents: List[List[str]],
		retrieved_metadata: List[List[Dict]],
	) -> List[str]:
		"""
		Replace retrieved_contents with a window to create a Prompt
		(only available for corpus chunked with Sentence window method)
		You must type a prompt or prompt list at a config YAML file like this:

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
		:return: Prompts that are made by window_replacement.
		"""

		def window_replacement_row(
			_prompt: str,
			_query: str,
			_retrieved_contents,
			_retrieved_metadata: List[Dict],
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
