import asyncio
from typing import List, Tuple

from llama_index.llms.base import BaseLLM
from transformers import AutoTokenizer

from autorag.nodes.generator.base import generator_node


@generator_node
def llama_index_llm(prompts: List[str], llm: BaseLLM) -> Tuple[List[str], List[List[int]], List[List[float]]]:
    """
    Llama Index LLM module.
    It gets the LLM instance from llama index, and returns generated text by the input prompt.
    It does not generate the right log probs, but it returns the pseudo log probs,
    which is not meant to be used for other modules.

    :return: A tuple of three elements.
        The first element is a list of generated text.
        The second element is a list of generated text's token ids, used tokenizer is GPT2Tokenizer.
        The third element is a list of generated text's pseudo log probs.
    """
    tasks = [llm.acomplete(prompt) for prompt in prompts]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))

    generated_texts = list(map(lambda x: x.text, results))
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    tokenized_ids = tokenizer(generated_texts).data['input_ids']
    pseudo_log_probs = list(map(lambda x: [0.5] * len(x), tokenized_ids))
    return generated_texts, tokenized_ids, pseudo_log_probs
