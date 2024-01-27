from llama_index.llms.openai import OpenAI

from autorag.nodes.queryexpansion import query_decompose

sample_query = ["Which group has more members, Newjeans or Espa?", "Which group has more members, STAYC or Espa?"]


def test_query_decompose():
    llm = OpenAI(temperature=0.2)
    result = query_decompose(sample_query, llm)
    assert len(result[0]) > 1
