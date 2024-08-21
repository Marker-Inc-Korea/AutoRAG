from typing import List, Callable, Dict

import pandas as pd

from autorag.nodes.queryexpansion.base import query_expansion_node

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


@query_expansion_node
def query_decompose(
	queries: List[str],
	generator_func: Callable,
	generator_params: Dict,
	prompt: str = decompose_prompt,
) -> List[List[str]]:
	"""
	decompose query to little piece of questions.
	:param queries: List[str], queries to decompose.
	:param generator_func: Callable, generator functions.
	:param generator_params: Dict, generator parameters.
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
	result_df = generator_func(
		project_dir=None, previous_result=input_df, **generator_params
	)
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
