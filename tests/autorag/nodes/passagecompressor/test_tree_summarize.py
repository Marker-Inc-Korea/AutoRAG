from typing import List

from llama_index.llms import OpenAI, MockLLM

from autorag.nodes.passagecompressor import tree_summarize

queries = [
    "What is the capital of France?",
    "What is the meaning of life?",
]
retrieved_contents = [
    ["Paris is the capital of France.", "France is a country in Europe.", "France is a member of the EU."],
    ["The meaning of life is 42.", "The meaning of life is to be happy.", "The meaning of life is to be kind."],
]


def check_result(result: List[str]):
    assert len(result) == len(queries)
    for r in result:
        assert isinstance(r, str)
        assert len(r) > 0
        assert bool(r) is True


def test_tree_summarize_default():
    mock = MockLLM()
    result = tree_summarize(queries, retrieved_contents, [], [], mock)
    check_result(result)


def test_tree_summarize_chat():
    gpt_3 = OpenAI(model_name="gpt-3.5-turbo")
    result = tree_summarize(queries, retrieved_contents, [], [], gpt_3)
    check_result(result)


def test_tree_summarize_custom_prompt():
    mock = MockLLM()
    prompt = "This is a custom prompt. {context_str} {query_str}"
    result = tree_summarize(queries, retrieved_contents, [], [], mock, prompt=prompt)
    check_result(result)
    assert 'This is a custom prompt.' in result[0]


def test_tree_summarize_custom_prompt_chat():
    gpt_3 = OpenAI(model_name="gpt-3.5-turbo")
    prompt = "Query: {query_str} Passages: {context_str}. Repeat the query."
    result = tree_summarize(queries, retrieved_contents, [], [], gpt_3, chat_prompt=prompt)
    check_result(result)
    assert 'What is the capital of France?' in result[0]
