from typing import List

import pandas as pd
from llama_index.llms import OpenAI, MockLLM

from autorag import generator_models
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
    result = tree_summarize.__wrapped__(queries, retrieved_contents, [], [], mock)
    check_result(result)


def test_tree_summarize_chat():
    gpt_3 = OpenAI(model="gpt-3.5-turbo")
    result = tree_summarize.__wrapped__(queries, retrieved_contents, [], [], gpt_3)
    check_result(result)


def test_tree_summarize_custom_prompt():
    mock = MockLLM()
    prompt = "This is a custom prompt. {context_str} {query_str}"
    result = tree_summarize.__wrapped__(queries, retrieved_contents, [], [], mock, prompt=prompt)
    check_result(result)
    assert 'This is a custom prompt.' in result[0]


def test_tree_summarize_custom_prompt_chat():
    gpt_3 = OpenAI(model="gpt-3.5-turbo")
    prompt = "Query: {query_str} Passages: {context_str}. Repeat the query."
    result = tree_summarize.__wrapped__(queries, retrieved_contents, [], [], gpt_3, chat_prompt=prompt)
    check_result(result)
    assert 'What is the capital of France?' in result[0]


def test_tree_summarize_node():
    generator_models['mock'] = MockLLM

    df = pd.DataFrame({
        'query': queries,
        'retrieved_contents': retrieved_contents,
        'retrieved_ids': [['id-1', 'id-2', 'id-3'], ['id-4', 'id-5', 'id-6']],
        'retrieve_scores': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    })
    result = tree_summarize(
        "project_dir",
        df,
        llm='mock',
        prompt="This is a custom prompt. {context_str} {query_str}",
        max_tokens=64,
    )
    assert isinstance(result, pd.DataFrame)
    contents = result['retrieved_contents'].tolist()
    assert isinstance(contents, list)
    assert len(contents) == len(queries)
    assert isinstance(contents[0], list)
    assert len(contents[0]) == 1
    assert isinstance(contents[0][0], str)
    assert bool(contents[0][0]) is True
