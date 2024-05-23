from autorag.nodes.queryexpansion import multi_query_expansion
from tests.autorag.nodes.queryexpansion.test_query_expansion_base import project_dir, previous_result, \
    base_query_expansion_node_test, ingested_vectordb_node
from tests.mock import MockLLM

sample_query = ["What are the potential benefits and drawbacks of implementing artificial intelligence in healthcare?",
                "How many members are in the group Aespa?"]


def test_multi_query_expansion():
    llm = MockLLM(temperature=0.2)
    original_multi_query_expansion = multi_query_expansion.__wrapped__
    result = original_multi_query_expansion(sample_query, llm)
    assert len(result[0]) > 1
    assert len(result) == 2
    assert isinstance(result[0][0], str)


def test_multi_query_expansion_node(ingested_vectordb_node):
    result_df = multi_query_expansion(project_dir=project_dir, previous_result=previous_result,
                                      llm='mock')
    base_query_expansion_node_test(result_df)
