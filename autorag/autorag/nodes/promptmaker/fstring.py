from typing import List

import pandas as pd

from autorag.nodes.promptmaker.base import BasePromptMaker
from autorag.utils import result_to_dataframe


class Fstring(BasePromptMaker):
	@result_to_dataframe(["prompts"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		query, retrieved_contents, prompt = self.cast_to_run(
			previous_result, *args, **kwargs
		)
		return self._pure(prompt, query, retrieved_contents)

	def _pure(
		self, prompt: str, queries: List[str], retrieved_contents: List[List[str]]
	) -> List[str]:
		"""
		Make a prompt using f-string from a query and retrieved_contents.
		You must type a prompt or prompt list at a config YAML file like this:

		.. Code:: yaml
		nodes:
		- node_type: prompt_maker
		  modules:
		  - module_type: fstring
			prompt: [Answer this question: {query} \n\n {retrieved_contents},
			Read the passages carefully and answer this question: {query} \n\n Passages: {retrieved_contents}]

		:param prompt: A prompt string.
		:param queries: List of query strings.
		:param retrieved_contents: List of retrieved contents.
		:return: Prompts that are made by f-string.
		"""

		def fstring_row(
			_prompt: str, _query: str, _retrieved_contents: List[str]
		) -> str:
			contents_str = "\n\n".join(_retrieved_contents)
			return _prompt.format(query=_query, retrieved_contents=contents_str)

		return list(
			map(
				lambda x: fstring_row(prompt, x[0], x[1]),
				zip(queries, retrieved_contents),
			)
		)
