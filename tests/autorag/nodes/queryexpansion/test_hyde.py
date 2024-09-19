import pytest

from autorag.nodes.queryexpansion import HyDE
from tests.autorag.nodes.queryexpansion.test_query_expansion_base import (
	project_dir,
	previous_result,
	base_query_expansion_node_test,
)

sample_query = ["How many members are in Newjeans?", "What is visconde structure?"]


@pytest.fixture
def hyde_instance():
	return HyDE(
		project_dir=project_dir, generator_module_type="llama_index_llm", llm="mock"
	)


def test_hyde(hyde_instance):
	result = hyde_instance._pure(sample_query, prompt="", temperature=0.1)
	assert len(result[0]) == 1
	assert len(result) == 2


def test_hyde_node():
	generator_dict = {"generator_module_type": "llama_index_llm", "llm": "mock"}
	result_df = HyDE.run_evaluator(
		project_dir=project_dir,
		previous_result=previous_result,
		**generator_dict,
		temperature=0.1,
	)
	base_query_expansion_node_test(result_df)
