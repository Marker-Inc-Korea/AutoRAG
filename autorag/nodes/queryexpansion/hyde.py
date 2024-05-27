from typing import List, Dict, Callable

import pandas as pd

from autorag.nodes.queryexpansion.base import query_expansion_node

hyde_prompt = "Please write a passage to answer the question"


@query_expansion_node
def hyde(queries: List[str],
         generator_func: Callable,
         generator_params: Dict,
         prompt: str = hyde_prompt,
         batch: int = 16) -> List[List[str]]:
    """
    HyDE, which inspired by "Precise Zero-shot Dense Retrieval without Relevance Labels" (https://arxiv.org/pdf/2212.10496.pdf)
    LLM model creates a hypothetical passage.
    And then, retrieve passages using hypothetical passage as a query.
    :param queries: List[str], queries to retrieve.
    :param generator_func: Callable, generator functions.
    :param generator_params: Dict, generator parameters.
    :param prompt: prompt to use when generating hypothetical passage
    :param batch: Batch size for llm.
        Default is 16.
    :return: List[List[str]], List of hyde results.
    """
    full_prompts = []
    for query in queries:
        if prompt is "":
            prompt = hyde_prompt
        full_prompt = prompt + f"\nQuestion: {query}\nPassage:"
        full_prompts.append(full_prompt)
    full_prompts = list(
        map(lambda x: (prompt if prompt else hyde_prompt) + f"\nQuestion: {x}\nPassage:", queries))
    input_df = pd.DataFrame({"prompts": full_prompts})
    result_df = generator_func(project_dir=None, previous_result=input_df, batch=batch, **generator_params)
    answers = result_df['generated_texts'].tolist()
    results = list(map(lambda x: [x], answers))
    return results
