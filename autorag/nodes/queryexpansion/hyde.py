import asyncio
from typing import List

from llama_index.llms.llm import BaseLLM

from autorag.nodes.queryexpansion.base import query_expansion_node
from autorag.utils.util import process_batch

hyde_prompt = "Please write a passage to answer the question"


@query_expansion_node
def hyde(queries: List[str], llm: BaseLLM,
         prompt: str = hyde_prompt,
         batch: int = 16) -> List[List[str]]:
    """
    HyDE, which inspired by "Precise Zero-shot Dense Retrieval without Relevance Labels" (https://arxiv.org/pdf/2212.10496.pdf)
    LLM model creates hypothetical passage.
    And then, retrieve passages using hypothetical passage as query.
    :param queries: List[str], queries to retrieve.
    :param llm: llm to use for hypothetical passage generation.
    :param prompt: prompt to use when generating hypothetical passage
    :param batch: Batch size for llm.
        Default is 16.
    :return: List[List[str]], List of hyde results.
    """
    # Run async query_decompose_pure function
    tasks = [hyde_pure(query, llm, prompt) for query in queries]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_batch(tasks, batch_size=batch))
    return results


async def hyde_pure(query: str, llm: BaseLLM,
                    prompt: str = hyde_prompt) -> List[str]:
    if prompt is "":
        prompt = hyde_prompt
    full_prompt = prompt + f"\nQuestion: {query}\nPassage:"
    hyde_answer = await llm.acomplete(full_prompt)
    return [hyde_answer.text]
