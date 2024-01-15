import asyncio
from typing import List

from llama_index.llms.llm import BaseLLM
from llama_index.llms.openai import OpenAI

hyde_prompt = "Please write a passage to answer the question"


def hyde(queries: List[str], llm: BaseLLM = OpenAI(max_tokens=64),
         prompt: str = hyde_prompt) -> List[List[str]]:
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
