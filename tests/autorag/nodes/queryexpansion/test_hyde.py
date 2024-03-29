from autorag import generator_models
from autorag.nodes.queryexpansion import hyde
from tests.autorag.nodes.queryexpansion.test_query_expansion_base import project_dir, previous_result, \
    base_query_expansion_node_test, ingested_vectordb_node
from tests.mock import MockLLM

sample_query = ["How many members are in Newjeans?", "What is visconde structure?"]


def test_hyde():
    llm = MockLLM()
    original_hyde = hyde.__wrapped__
    result = original_hyde(sample_query, llm, prompt="")
    assert len(result[0]) == 1
    assert len(result) == 2


def test_hyde_node(ingested_vectordb_node):
    generator_models['mock'] = MockLLM
    result_df = hyde(project_dir=project_dir, previous_result=previous_result,
                     llm="mock", max_tokens=64)
    base_query_expansion_node_test(result_df)
