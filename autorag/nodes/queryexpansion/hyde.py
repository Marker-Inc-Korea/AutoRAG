from typing import List

import pandas as pd

from autorag.nodes.queryexpansion.base import BaseQueryExpansion
from autorag.utils import result_to_dataframe

hyde_prompt = "Please write a passage to answer the question"


class HyDE(BaseQueryExpansion):
	@result_to_dataframe(["queries"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries = self.cast_to_run(previous_result, *args, **kwargs)

		# pop prompt from kwargs
		prompt = kwargs.pop("prompt", hyde_prompt)
		kwargs.pop("generator_module_type", None)

		expanded_queries = self._pure(queries, prompt, **kwargs)
		return self._check_expanded_query(queries, expanded_queries)

	def _pure(self, queries: List[str], prompt: str = hyde_prompt, **generator_params):
		"""
		HyDE, which inspired by "Precise Zero-shot Dense Retrieval without Relevance Labels" (https://arxiv.org/pdf/2212.10496.pdf)
		LLM model creates a hypothetical passage.
		And then, retrieve passages using hypothetical passage as a query.
		:param queries: List[str], queries to retrieve.
		:param prompt: Prompt to use when generating hypothetical passage
		:return: List[List[str]], List of hyde results.
		"""
		full_prompts = list(
			map(
				lambda x: (prompt if not bool(prompt) else hyde_prompt)
				+ f"\nQuestion: {x}\nPassage:",
				queries,
			)
		)
		input_df = pd.DataFrame({"prompts": full_prompts})
		result_df = self.generator.pure(previous_result=input_df, **generator_params)
		answers = result_df["generated_texts"].tolist()
		results = list(map(lambda x: [x], answers))
		return results
