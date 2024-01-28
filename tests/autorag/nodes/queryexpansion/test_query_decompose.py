from llama_index.llms.openai import OpenAI

from autorag.nodes.queryexpansion import query_decompose
from tests.autorag.nodes.queryexpansion.test_query_expansion_base import project_dir, previous_result, \
    base_query_expansion_node_test, ingested_vectordb_node


sample_query = ["Which group has more members, Newjeans or Espa?", "Which group has more members, STAYC or Espa?"]


def test_query_decompose():
    llm = OpenAI(temperature=0.2)
    original_query_decompose = query_decompose.__wrapped__
    result = original_query_decompose(sample_query, llm, prompt="")
    assert len(result[0]) > 1
    assert len(result) == 2
    assert isinstance(result[0][0], str)


def test_query_decompose_node(ingested_vectordb_node):
    result_df = query_decompose(project_dir=project_dir, previous_result=previous_result,
                                llm="openai", temperature=0.2)
    base_query_expansion_node_test(result_df)
