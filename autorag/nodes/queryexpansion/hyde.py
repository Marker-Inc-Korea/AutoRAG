import asyncio
from typing import List

from llama_index.llms.llm import BaseLLM
from llama_index.llms.openai import OpenAI

hyde_prompt = "Please write a passage to answer the question"


def hyde(queries: List[str], llm: BaseLLM = OpenAI(max_tokens=64),
         prompt: str = hyde_prompt) -> List[List[str]]:
    """
    HyDE, which inspired by "Precise Zero-shot Dense Retrieval without Relevance Labels" (https://arxiv.org/pdf/2212.10496.pdf)
    LLM model creates hypothetical passage.
    And then, retrieve passages using hypothetical passage as query.
    :param queries: List[str], queries to retrieve.
    :param llm: llm to use for hypothetical passage generation. HyDE Retrieval supports both chat and completion LLMs.
    :param prompt: prompt to use when generating hypothetical passage
    :return: List[List[str]], List of retrieved passages
    """
    # Run async query_decompose_pure function
    tasks = [hyde_pure(query, llm, prompt) for query in queries]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    return results


async def hyde_pure(query: str, llm: BaseLLM,
                    prompt: str = hyde_prompt) -> List[str]:
    full_prompt = prompt + f"\nQuestion: {query}\nPassage:"
    hyde_answer = llm.complete(full_prompt)
    return [hyde_answer.text]
