import pytest

from autorag.nodes.promptmaker.chat_fstring import ChatFstring
from tests.autorag.nodes.promptmaker.test_prompt_maker_base import (
    queries,
    retrieved_contents,
    previous_result,
)


@pytest.fixture
def fstring_instance():
    return ChatFstring("pseudo_project_dir")


prompt = [
    {
        "role": "system",
        "content": "You are a helpful assistant that helps people find information.",
    },
    {"role": "user", "content": "Answer this question: {query}\n{retrieved_contents}"},
]


def test_fstring(fstring_instance):
    result_prompts = fstring_instance._pure(prompt, queries, retrieved_contents)
    assert len(result_prompts) == 2
    assert isinstance(result_prompts, list)
    assert isinstance(result_prompts[0], list)
    assert isinstance(result_prompts[0][0], dict)
    assert isinstance(result_prompts[0][1], dict)
    assert result_prompts[0] == [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": "Answer this question: What is the capital of Japan?\nTokyo is the capital of Japan.\n\nTokyo, the capital of Japan, is a huge metropolitan city.",
        },
    ]
    assert result_prompts[1] == [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": "Answer this question: What is the capital of China?\nBeijing is the capital of China.\n\nBeijing, the capital of China, is a huge metropolitan city.",
        },
    ]


def test_fstring_node():
    result = ChatFstring.run_evaluator(
        project_dir="pseudo_project_dir", previous_result=previous_result, prompt=prompt
    )
    assert len(result) == 2
    assert result.columns == ["prompts"]
    assert result["prompts"][0] == [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": "Answer this question: What is the capital of Japan?\nTokyo is the capital of Japan.\n\nTokyo, the capital of Japan, is a huge metropolitan city.",
        },
    ]
    assert result["prompts"][1] == [
        {
            "role": "system",
            "content": "You are a helpful assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": "Answer this question: What is the capital of China?\nBeijing is the capital of China.\n\nBeijing, the capital of China, is a huge metropolitan city.",
        },
    ]
