import asyncio
from typing import List

from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType

from autorag.nodes.queryexpansion.base import query_expansion_node
from autorag.utils.util import process_batch

multi_query_expansion_prompt = """You are an AI language model assistant. 
    Your task is to generate 3 different versions of the given user 
    question to retrieve relevant documents from a vector  database. 
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations 
    of distance-based similarity search. Provide these alternative 
    questions separated by newlines. Original question: {question}"""


@query_expansion_node
def multi_query_expansion(queries: List[str], llm: LLMPredictorType,
                          prompt: str = multi_query_expansion_prompt,
                          batch: int = 16) -> List[List[str]]:
    """
    Expand a list of queries using a multi-query expansion approach.
    LLM model generate 3 different versions queries for each input query.

    :param queries: List[str], queries to decompose.
    :param llm: LLMPredictorType, language model to use.
    :param prompt: str, prompt to use for multi-query expansion.
        default prompt comes from langchain MultiQueryRetriever default query prompt.
    :param batch: int, batch size for llm.
        Default is 16.
    :return: List[List[str]], list of expansion query.
    """
    tasks = [multi_query_expansion_pure(query, llm, prompt) for query in queries]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_batch(tasks, batch_size=batch))
    return results


async def multi_query_expansion_pure(query: str, llm: LLMPredictorType,
                                     prompt: str = multi_query_expansion_prompt) -> List[str]:
    if prompt == "":
        prompt = multi_query_expansion_prompt
    full_prompt = prompt.format(question=query)
    answer = await llm.acomplete(full_prompt)
    try:
        queries = answer.text.split("\n")
        queries.insert(0, query)
        return queries
    except:
        return [query]
