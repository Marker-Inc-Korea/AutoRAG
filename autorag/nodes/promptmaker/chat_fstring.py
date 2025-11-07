import copy
from typing import List

import pandas as pd

from autorag.nodes.promptmaker.base import BasePromptMaker
from autorag.utils import result_to_dataframe


class ChatFstring(BasePromptMaker):
	@result_to_dataframe(["prompts"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		query, retrieved_contents, prompt = self.cast_to_run(
			previous_result, *args, **kwargs
		)
		return self._pure(prompt, query, retrieved_contents)

	def _pure(
		self,
		prompt: List[dict[str, str]],
		queries: List[str],
		retrieved_contents: List[List[str]],
	) -> List[List[dict[str, str]]]:
		"""
		Make a prompt using f-string from a query and retrieved_contents.
		You must type a prompt or prompt list at a config YAML file like this:

		.. Code:: yaml
		nodes:
		- node_type: prompt_maker
		  modules:
		  - module_type: chatfstring
			prompt:
			- - role: system
			  content: You are a helpful assistant that helps people find information.
			  - role: user
			    content: |
			    Answer this question: {query}
			    {retrieved_contents}
			- - role: system
			    content: You are helpful.
			  - role: user
				content: |
			    Read the passages carefully and answer this question: {query}

			    Passages: {retrieved_contents}

		:param prompt: A prompt string.
		:param queries: List of query strings.
		:param retrieved_contents: List of retrieved contents.
		:return: Prompts that are made by chat f-string.
			It is the list of OpenAI chat format prompts.
		"""

		def fstring_row(
			_prompt: List[dict[str, str]], _query: str, _retrieved_contents: List[str]
		) -> List[dict[str, str]]:
			contents_str = "\n\n".join(_retrieved_contents)
			result_prompt = copy.deepcopy(_prompt)
			for lst in result_prompt:
				if "content" in lst:
					lst["content"] = lst["content"].format(
						query=_query, retrieved_contents=contents_str
					)

			return result_prompt

		return list(
			map(
				lambda x: fstring_row(prompt, x[0], x[1]),
				zip(queries, retrieved_contents),
			)
		)
