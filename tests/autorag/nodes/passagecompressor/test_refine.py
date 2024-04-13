import pandas as pd

from autorag import generator_models
from autorag.nodes.passagecompressor.refine import refine
from tests.autorag.nodes.passagecompressor.test_base_passage_compressor import (queries, retrieved_contents,
                                                                                check_result, df)
from tests.mock import MockLLM


def test_refine_default():
    llm = MockLLM()
    result = refine.__wrapped__(queries, retrieved_contents, [], [], llm)
    check_result(result)


def test_refine_chat():
    gpt_3 = MockLLM(model="gpt-3.5-turbo")
    result = refine.__wrapped__(queries, retrieved_contents, [], [], gpt_3)
    check_result(result)


def test_refine_custom_prompt():
    llm = MockLLM()
    prompt = "This is a custom prompt. {context_msg} {query_str}"
    result = refine.__wrapped__(queries, retrieved_contents, [], [], llm, prompt=prompt)
    check_result(result)
    assert bool(result[0]) is True


def test_refine_custom_prompt_chat():
    gpt_3 = MockLLM(model="gpt-3.5-turbo")
    prompt = "Query: {query_str} Passages: {context_msg}. Repeat the query."
    result = refine.__wrapped__(queries, retrieved_contents, [], [], gpt_3, chat_prompt=prompt)
    check_result(result)
    assert 'What is the capital of France?' in result[0]


def test_refine_node():
    generator_models['mock'] = MockLLM
    result = refine(
        "project_dir",
        df,
        llm='mock',
        prompt="This is a custom prompt. {context_msg} {query_str}",
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
