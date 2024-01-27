from llama_index.llms.openai import OpenAI

from autorag.nodes.queryexpansion import hyde

sample_query = ["How many members are in Newjeans?", "What is visconde structure?"]


def test_hyde():
    llm = OpenAI(max_tokens=64)
    result = hyde(sample_query, llm)
    assert len(result[0]) == 1
