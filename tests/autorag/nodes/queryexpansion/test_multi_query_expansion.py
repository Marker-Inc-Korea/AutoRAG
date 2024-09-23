import pytest

from autorag.nodes.queryexpansion import MultiQueryExpansion
from tests.autorag.nodes.queryexpansion.test_query_expansion_base import (
	project_dir,
	previous_result,
	base_query_expansion_node_test,
)

sample_query = [
	"What are the potential benefits and drawbacks of implementing artificial intelligence in healthcare?",
	"How many members are in the group Aespa?",
]


@pytest.fixture
def multi_query_expansion_instance():
	return MultiQueryExpansion(
		project_dir=project_dir, generator_module_type="llama_index_llm", llm="mock"
	)


def test_multi_query_expansion(multi_query_expansion_instance):
	result = multi_query_expansion_instance._pure(
		sample_query,
		prompt="Generate Multi Query Expansion : {query}",
	)
	assert len(result[0]) > 1
	assert len(result) == 2
	assert isinstance(result[0][0], str)


def test_multi_query_expansion_node():
	generator_dict = {"generator_module_type": "llama_index_llm", "llm": "mock"}
	result_df = MultiQueryExpansion.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, **generator_dict
	)
	base_query_expansion_node_test(result_df)
