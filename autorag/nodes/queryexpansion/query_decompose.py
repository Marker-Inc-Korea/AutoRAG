import asyncio
from typing import List

from llama_index.llms.llm import BaseLLM

from autorag.nodes.queryexpansion.base import query_expansion_node
from autorag.utils.util import process_batch

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

    Decompositions:"
    """


@query_expansion_node
def query_decompose(queries: List[str], llm: BaseLLM,
                    prompt: str = decompose_prompt,
                    batch: int = 16) -> List[List[str]]:
    """
    decompose query to little piece of questions.
    :param queries: List[str], queries to decompose.
    :param llm: BaseLLM, language model to use.
    :param prompt: str, prompt to use for query decomposition.
        default prompt comes from Visconde's StrategyQA few-shot prompt.
    :param batch: int, batch size for llm.
        Default is 16.
    :return: List[List[str]], list of decomposed query. Return input query if query is not decomposable.
    """
    # Run async query_decompose_pure function
    tasks = [query_decompose_pure(query, llm, prompt) for query in queries]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_batch(tasks, batch_size=batch))
    return results


async def query_decompose_pure(query: str, llm: BaseLLM,
                               prompt: str = decompose_prompt) -> List[str]:
    """
    decompose query to little piece of questions.
    :param query: str, query to decompose.
    :param llm: BaseLLM, language model to use.
    :param prompt: str, prompt to use for query decomposition.
        default prompt comes from Visconde's StrategyQA few-shot prompt.
    :return: List[str], list of decomposed query. Return input query if query is not decomposable.
    """
    if prompt == "":
        prompt = decompose_prompt
    full_prompt = "prompt: " + prompt + "\n\n" "question: " + query
    answer = await llm.acomplete(full_prompt)
    if answer.text == "the question needs no decomposition.":
        return [query]
    try:
        lines = [line.strip() for line in answer.text.splitlines() if line.strip()]
        if lines[0].startswith("Decompositions:"):
            lines.pop(0)
        questions = [line.split(':', 1)[1].strip() for line in lines if ':' in line]
        if not questions:
            return [query]
        return questions
    except:
        return [query]
