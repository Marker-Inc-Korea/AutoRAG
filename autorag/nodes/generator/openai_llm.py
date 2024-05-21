import asyncio
import logging
import os
from typing import List, Tuple

import tiktoken
from openai import AsyncOpenAI
from tiktoken import Encoding

from autorag.nodes.generator.base import generator_node
from autorag.utils.util import process_batch

logger = logging.getLogger("AutoRAG")

MAX_TOKEN_DICT = {  # model name : token limit
    'gpt-4o': 128_000,
    'gpt-4o-2024-05-13': 128_000,
    'gpt-4-turbo': 128_000,
    'gpt-4-turbo-2024-04-09': 128_000,
    'gpt-4-turbo-preview': 128_000,
    'gpt-4-0125-preview': 128_000,
    'gpt-4-1106-preview': 128_000,
    'gpt-4-vision-preview': 128_000,
    'gpt-4-1106-vision-preview': 128_000,
    'gpt-4': 8_192,
    'gpt-4-0613': 8_192,
    'gpt-4-32k': 32_768,
    'gpt-4-32k-0613': 32_768,
    'gpt-3.5-turbo-0125': 16_385,
    'gpt-3.5-turbo': 16_385,
    'gpt-3.5-turbo-1106': 16_385,
    'gpt-3.5-turbo-instruct': 4_096,
    'gpt-3.5-turbo-16k': 16_385,
    'gpt-3.5-turbo-0613': 4_096,
    'gpt-3.5-turbo-16k-0613': 16_385,
}


@generator_node
def openai_llm(prompts: List[str], llm: str = "gpt-3.5-turbo", batch: int = 16,
               truncate: bool = True,
               api_key: str = None,
               **kwargs) -> \
        Tuple[List[str], List[List[int]], List[List[float]]]:
    """
    OpenAI generator module.
    Uses official openai library for generating answer from the given prompt.
    It returns real token ids and log probs, so you must use this for using token ids and log probs.

    :param prompts: A list of prompts.
    :param llm: A model name for openai.
        Default is gpt-3.5-turbo.
    :param batch: Batch size for openai api call.
        If you get API limit errors, you should lower the batch size.
        Default is 16.
    :param truncate: Whether to truncate the input prompt.
        Default is True.
    :param api_key: OpenAI API key. You can set this by passing env variable `OPENAI_API_KEY`
    :param kwargs: The optional parameter for openai api call `openai.chat.completion`
        See https://platform.openai.com/docs/api-reference/chat/create for more details.
    :return: A tuple of three elements.
        The first element is a list of generated text.
        The second element is a list of generated text's token ids.
        The third element is a list of generated text's log probs.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY does not set. "
                             "Please set env variable OPENAI_API_KEY or pass api_key parameter to openai module.")

    if kwargs.get('logprobs') is not None:
        kwargs.pop('logprobs')
        logger.warning("parameter logprob does not effective. It always set to True.")
    if kwargs.get('n') is not None:
        kwargs.pop('n')
        logger.warning("parameter n does not effective. It always set to 1.")

    tokenizer = tiktoken.encoding_for_model(llm)
    if truncate:
        max_token_size = MAX_TOKEN_DICT.get(llm) - 7  # because of chat token usage
        if max_token_size is None:
            raise ValueError(f"Model {llm} does not supported. "
                             f"Please select the model between {list(MAX_TOKEN_DICT.keys())}")
        prompts = list(map(lambda prompt: truncate_by_token(prompt, tokenizer, max_token_size), prompts))

    client = AsyncOpenAI(api_key=api_key)
    loop = asyncio.get_event_loop()
    tasks = [get_result(prompt, client, llm, tokenizer, **kwargs) for prompt in prompts]
    result = loop.run_until_complete(process_batch(tasks, batch))
    answer_result = list(map(lambda x: x[0], result))
    token_result = list(map(lambda x: x[1], result))
    logprob_result = list(map(lambda x: x[2], result))
    return answer_result, token_result, logprob_result


async def get_result(prompt: str, client: AsyncOpenAI, model: str, tokenizer: Encoding, **kwargs):
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        logprobs=True,
        n=1,
        **kwargs
    )
    choice = response.choices[0]
    answer = choice.message.content
    logprobs = list(map(lambda x: x.logprob, choice.logprobs.content))
    tokens = tokenizer.encode(answer, allowed_special='all')
    assert len(tokens) == len(logprobs), "tokens and logprobs size is different."
    return answer, tokens, logprobs


def truncate_by_token(prompt: str, tokenizer: Encoding, max_token_size: int):
    tokens = tokenizer.encode(prompt, allowed_special='all')
    return tokenizer.decode(tokens[:max_token_size])
