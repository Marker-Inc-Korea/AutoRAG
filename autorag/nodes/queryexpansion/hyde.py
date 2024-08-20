from typing import List, Dict, Callable

import pandas as pd

from autorag.nodes.queryexpansion.base import query_expansion_node

hyde_prompt = "Please write a passage to answer the question"


@query_expansion_node
def hyde(
	queries: List[str],
	generator_func: Callable,
	generator_params: Dict,
	prompt: str = hyde_prompt,
) -> List[List[str]]:
	"""
	HyDE, which inspired by "Precise Zero-shot Dense Retrieval without Relevance Labels" (https://arxiv.org/pdf/2212.10496.pdf)
	LLM model creates a hypothetical passage.
	And then, retrieve passages using hypothetical passage as a query.
	:param queries: List[str], queries to retrieve.
	:param generator_func: Callable, generator functions.
	:param generator_params: Dict, generator parameters.
	:param prompt: prompt to use when generating hypothetical passage
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
	result_df = generator_func(
		project_dir=None, previous_result=input_df, **generator_params
	)
	answers = result_df["generated_texts"].tolist()
	results = list(map(lambda x: [x], answers))
	return results
