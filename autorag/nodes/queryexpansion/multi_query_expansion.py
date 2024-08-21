from typing import List, Dict, Callable

import pandas as pd

from autorag.nodes.queryexpansion.base import query_expansion_node

multi_query_expansion_prompt = """You are an AI language model assistant.
    Your task is to generate 3 different versions of the given user
    question to retrieve relevant documents from a vector  database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. Provide these alternative
    questions separated by newlines. Original question: {question}"""


@query_expansion_node
def multi_query_expansion(
	queries: List[str],
	generator_func: Callable,
	generator_params: Dict,
	prompt: str = multi_query_expansion_prompt,
) -> List[List[str]]:
	"""
	Expand a list of queries using a multi-query expansion approach.
	LLM model generate 3 different versions queries for each input query.

	:param queries: List[str], queries to decompose.
	:param generator_func: Callable, generator functions.
	:param generator_params: Dict, generator parameters.
	:param prompt: str, prompt to use for multi-query expansion.
	    default prompt comes from langchain MultiQueryRetriever default query prompt.
	:return: List[List[str]], list of expansion query.
	"""
	full_prompts = []
	for query in queries:
		if bool(prompt):
			full_prompt = f"prompt: {prompt}\n\n question: {query}"
		else:
			full_prompt = multi_query_expansion_prompt.format(question=query)
		full_prompts.append(full_prompt)
	input_df = pd.DataFrame({"prompts": full_prompts})
	result_df = generator_func(
		project_dir=None, previous_result=input_df, **generator_params
	)
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
