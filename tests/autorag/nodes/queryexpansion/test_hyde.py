from llama_index.llms.openai import OpenAI

from autorag.nodes.queryexpansion import hyde

sample_query = ["How many members are in Newjeans?", "What is visconde structure?"]


def test_hyde():
    llm = OpenAI(max_tokens=64)
    original_hyde = hyde.__wrapped__
    result = original_hyde(sample_query, llm, prompt=None)
    assert len(result[0]) == 1
    assert len(result) == 2


def test_hyde_node():
    pass
