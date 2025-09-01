import pandas as pd
from vllm_mock import LLM

from autorag.nodes.generator import Vllm
from tests.autorag.nodes.generator.test_generator_base import (
    prompts,
    chat_prompts,
    check_generated_texts,
    check_generated_tokens,
    check_generated_log_probs,
    check_generated_texts_chat,
    check_generated_tokens_chat,
    check_generated_log_probs_chat,
)


def test_vllm(mocker):
    mock_class = mocker.patch("vllm.LLM")
    mock_class.return_value = LLM(model="mock-model")

    previous_result = pd.DataFrame(
        {"prompts": prompts, "qid": ["id-1", "id-2", "id-3"]}
    )
    result_df = Vllm.run_evaluator(
        project_dir=".",
        previous_result=previous_result,
        tensor_parallel_size=1,
        llm="facebook/opt-125m",
        max_tokens=5,
        temperature=0.5,
    )
    tokens = result_df["generated_tokens"].tolist()
    log_probs = result_df["generated_log_probs"].tolist()
    check_generated_texts(result_df["generated_texts"].tolist())
    check_generated_tokens(tokens)
    check_generated_log_probs(log_probs)
    assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))


def test_vllm_chat_prompt(mocker):
    mock_class = mocker.patch("vllm.LLM")
    mock_class.return_value = LLM(model="mock-model")

    previous_result = pd.DataFrame(
        {"prompts": chat_prompts, "qid": ["id-1", "id-2", "id-3"]}
    )
    result_df = Vllm.run_evaluator(
        project_dir=".",
        previous_result=previous_result,
        tensor_parallel_size=1,
        llm="facebook/opt-125m",
        max_tokens=5,
        temperature=0.5,
    )
    tokens = result_df["generated_tokens"].tolist()
    log_probs = result_df["generated_log_probs"].tolist()
    check_generated_texts_chat(result_df["generated_texts"].tolist())
    check_generated_tokens_chat(tokens)
    check_generated_log_probs_chat(log_probs)
    assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))


def test_vllm_chat_prompt_think(mocker):
    mock_class = mocker.patch("vllm.LLM")
    mock_class.return_value = LLM(model="mock-model")

    previous_result = pd.DataFrame(
        {"prompts": chat_prompts, "qid": ["id-1", "id-2", "id-3"]}
    )
    result_df = Vllm.run_evaluator(
        project_dir=".",
        previous_result=previous_result,
        tensor_parallel_size=1,
        llm="facebook/opt-125m",
        max_tokens=5,
        temperature=0.5,
        thinking=True,
    )
    tokens = result_df["generated_tokens"].tolist()
    log_probs = result_df["generated_log_probs"].tolist()
    check_generated_texts_chat(result_df["generated_texts"].tolist())
    check_generated_tokens_chat(tokens)
    check_generated_log_probs_chat(log_probs)
    assert all(len(tokens[i]) == len(log_probs[i]) for i in range(len(tokens)))
