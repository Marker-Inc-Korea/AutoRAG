from typing import List

import pandas as pd

from autorag.nodes.queryexpansion.base import BaseQueryExpansion
from autorag.utils import result_to_dataframe

multi_query_expansion_prompt = """You are an AI language model assistant.
    Your task is to generate 3 different versions of the given user
    question to retrieve relevant documents from a vector  database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. Provide these alternative
    questions separated by newlines. Original question: {query}"""


class MultiQueryExpansion(BaseQueryExpansion):
	@result_to_dataframe(["queries"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries = self.cast_to_run(previous_result, *args, **kwargs)

		# pop prompt from kwargs
		prompt = kwargs.pop("prompt", multi_query_expansion_prompt)
		kwargs.pop("generator_module_type", None)

		expanded_queries = self._pure(queries, prompt, **kwargs)
		return self._check_expanded_query(queries, expanded_queries)

	def _pure(
		self, queries, prompt: str = multi_query_expansion_prompt, **kwargs
	) -> List[List[str]]:
		"""
		Expand a list of queries using a multi-query expansion approach.
		LLM model generate 3 different versions queries for each input query.

		:param queries: List[str], queries to decompose.
		:param prompt: str, prompt to use for multi-query expansion.
			default prompt comes from langchain MultiQueryRetriever default query prompt.
		:return: List[List[str]], list of expansion query.
		"""
		full_prompts = list(map(lambda x: prompt.format(query=x), queries))
		input_df = pd.DataFrame({"prompts": full_prompts})
		result_df = self.generator.pure(previous_result=input_df, **kwargs)
		answers = result_df["generated_texts"].tolist()
		results = list(
			map(lambda x: get_multi_query_expansion(x[0], x[1]), zip(queries, answers))
		)
		return results


def get_multi_query_expansion(query: str, answer: str) -> List[str]:
	try:
		queries = answer.split("\n")
		queries.insert(0, query)
		return queries
	except:
		return [query]
