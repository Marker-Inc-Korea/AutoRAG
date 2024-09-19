import pytest

from autorag.nodes.queryexpansion import QueryDecompose
from tests.autorag.nodes.queryexpansion.test_query_expansion_base import (
	project_dir,
	previous_result,
	base_query_expansion_node_test,
)

sample_query = [
	"Which group has more members, Newjeans or Aespa?",
	"Which group has more members, STAYC or Aespa?",
]


@pytest.fixture
def query_decompose_instance():
	return QueryDecompose(
		project_dir=project_dir, generator_module_type="llama_index_llm", llm="mock"
	)


def test_query_decompose(query_decompose_instance):
	result = query_decompose_instance._pure(sample_query)
	assert len(result[0]) > 1
	assert len(result) == 2
	assert isinstance(result[0][0], str)


def test_query_decompose_node():
	generator_dict = {"generator_module_type": "llama_index_llm", "llm": "mock"}
	result_df = QueryDecompose.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, **generator_dict
	)
	base_query_expansion_node_test(result_df)
