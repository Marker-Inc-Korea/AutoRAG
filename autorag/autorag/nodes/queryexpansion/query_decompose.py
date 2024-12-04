from typing import List

import pandas as pd

from autorag.nodes.queryexpansion.base import BaseQueryExpansion
from autorag.utils import result_to_dataframe

decompose_prompt = """Decompose a question in self-contained sub-questions. Use \"The question needs no decomposition\" when no decomposition is needed.

    Example 1:

    Question: Is Hamlet more common on IMDB than Comedy of Errors?
    Decompositions:
    1: How many listings of Hamlet are there on IMDB?
    2: How many listing of Comedy of Errors is there on IMDB?

    Example 2:

    Question: Are birds important to badminton?

    Decompositions:
    The question needs no decomposition

    Example 3:

    Question: Is it legal for a licensed child driving Mercedes-Benz to be employed in US?

    Decompositions:
    1: What is the minimum driving age in the US?
    2: What is the minimum age for someone to be employed in the US?

    Example 4:

    Question: Are all cucumbers the same texture?

    Decompositions:
    The question needs no decomposition

    Example 5:

    Question: Hydrogen's atomic number squared exceeds number of Spice Girls?

    Decompositions:
    1: What is the atomic number of hydrogen?
    2: How many Spice Girls are there?

    Example 6:

    Question: {question}

    Decompositions:
    """


class QueryDecompose(BaseQueryExpansion):
	@result_to_dataframe(["queries"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries = self.cast_to_run(previous_result, *args, **kwargs)

		# pop prompt from kwargs
		prompt = kwargs.pop("prompt", decompose_prompt)
		kwargs.pop("generator_module_type", None)

		expanded_queries = self._pure(queries, prompt, **kwargs)
		return self._check_expanded_query(queries, expanded_queries)

	def _pure(
		self, queries: List[str], prompt: str = decompose_prompt, *args, **kwargs
	) -> List[List[str]]:
		"""
		decompose query to little piece of questions.
		:param queries: List[str], queries to decompose.
		:param prompt: str, prompt to use for query decomposition.
			default prompt comes from Visconde's StrategyQA few-shot prompt.
		:return: List[List[str]], list of decomposed query. Return input query if query is not decomposable.
		"""
		full_prompts = []
		for query in queries:
			if bool(prompt):
				full_prompt = f"prompt: {prompt}\n\n question: {query}"
			else:
				full_prompt = decompose_prompt.format(question=query)
			full_prompts.append(full_prompt)
		input_df = pd.DataFrame({"prompts": full_prompts})
		result_df = self.generator.pure(previous_result=input_df, *args, **kwargs)
		answers = result_df["generated_texts"].tolist()
		results = list(
			map(lambda x: get_query_decompose(x[0], x[1]), zip(queries, answers))
		)
		return results


def get_query_decompose(query: str, answer: str) -> List[str]:
	"""
	decompose query to little piece of questions.
	:param query: str, query to decompose.
	:param answer: str, answer from query_decompose function.
	:return: List[str], list of a decomposed query. Return input query if query is not decomposable.
	"""
	if answer.lower() == "the question needs no decomposition":
		return [query]
	try:
		lines = [line.strip() for line in answer.splitlines() if line.strip()]
		if lines[0].startswith("Decompositions:"):
			lines.pop(0)
		questions = [line.split(":", 1)[1].strip() for line in lines if ":" in line]
		if not questions:
			return [query]
		return questions
	except:
		return [query]
