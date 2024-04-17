import asyncio
from typing import List, Tuple

from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType

from autorag.nodes.generator.base import generator_node, get_tokens_log
from autorag.utils.util import process_batch


@generator_node
def llama_index_llm(prompts: List[str], llm: LLMPredictorType, batch: int = 16) -> Tuple[List[str], List[List[int]], List[List[float]]]:
    """
    Llama Index LLM module.
    It gets the LLM instance from llama index, and returns generated text by the input prompt.
    It does not generate the right log probs, but it returns the pseudo log probs,
    which is not meant to be used for other modules.

    :param prompts: A list of prompts.
    :param llm: A llama index LLM instance.
    :param batch: The batch size for llm.
        Set low if you face some errors.
    :return: A tuple of three elements.
        The first element is a list of generated text.
        The second element is a list of generated text's token ids, used tokenizer is GPT2Tokenizer.
        The third element is a list of generated text's pseudo log probs.
    """
    tasks = [llm.acomplete(prompt) for prompt in prompts]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_batch(tasks, batch_size=batch))

    generated_texts = list(map(lambda x: x.text, results))
    tokenized_ids, pseudo_log_probs = get_tokens_log(generated_texts)
    return generated_texts, tokenized_ids, pseudo_log_probs
