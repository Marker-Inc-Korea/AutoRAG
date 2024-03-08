import pytest

from autorag.nodes.generator import vllm
from tests.autorag.nodes.generator.test_generator_base import (prompts, check_generated_texts, check_generated_tokens,
                                                               check_generated_log_probs)


@pytest.mark.skip(reason="vllm have to run with CUDA supported device.")
def test_vllm():
    answers, tokens, log_probs = vllm(
        prompts, llm='facebook/opt-125m', max_tokens=5, temperature=0.5,
    )
    check_generated_texts(answers)
    check_generated_tokens(tokens)
    check_generated_log_probs(log_probs)
    assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))
