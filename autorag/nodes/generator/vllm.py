import inspect
import itertools
from copy import deepcopy
from typing import List, Tuple

import torch

from autorag.nodes.generator.base import generator_node


@generator_node
def vllm(prompts: List[str], llm: str, **kwargs) -> Tuple[List[str], List[List[int]], List[List[float]]]:
    """
    Vllm module.
    It gets the VLLM instance, and returns generated texts by the input prompt.

    :param prompts: A list of prompts.
    :param llm: Model name of vLLM.
    :param kwargs: The extra parameters for generating the text.
    :return: A tuple of three elements.
        The first element is a list of generated text.
        The second element is a list of generated text's token ids.
        The third element is a list of generated text's log probs.
    """
    try:
        from vllm.outputs import RequestOutput
        from vllm.sequence import SampleLogprobs
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("Please install vllm library. You can install it by running `pip install vllm`.")

    input_kwargs = deepcopy(kwargs)
    vllm_model = make_vllm_instance(llm, input_kwargs)

    generate_params = SamplingParams(**input_kwargs)
    results: List[RequestOutput] = vllm_model.generate(prompts, generate_params)
    flatten_outputs = list(itertools.chain.from_iterable(list(map(lambda x: x.outputs, results))))
    generated_texts = list(map(lambda x: x.text, flatten_outputs))
    generated_token_ids = list(map(lambda x: x.token_ids, flatten_outputs))
    log_probs: List[SampleLogprobs] = list(map(lambda x: x.log_probs, flatten_outputs))
    generated_log_probs = list(map(lambda x: list(map(
        lambda y: y[0].logprob, x
    )), log_probs))
    destroy_vllm_instance(vllm_model)
    return generated_texts, generated_token_ids, generated_log_probs


def make_vllm_instance(llm: str, input_args):
    from vllm import LLM
    model_from_args = input_args.pop('model', None)
    model = llm if model_from_args is None else model_from_args
    init_params = inspect.signature(LLM.__init__).parameters.values()
    keyword_init_params = [param.name for param in init_params if param.kind == param.KEYWORD_ONLY]
    input_kwargs = {}
    for param in keyword_init_params:
        v = input_args.pop(param, None)
        if v is not None:
            input_kwargs[param] = v
    return LLM(model, **input_kwargs)


def destroy_vllm_instance(vllm_instance):
    if torch.cuda.is_available():
        from vllm.model_executor.parallel_utils.parallel_state import (
            destroy_model_parallel,
        )

        destroy_model_parallel()
        del vllm_instance
        torch.cuda.synchronize()
    else:
        del vllm_instance
