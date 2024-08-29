from autorag.nodes.queryexpansion import hyde
from autorag.support import get_support_modules
from tests.autorag.nodes.queryexpansion.test_query_expansion_base import (
	project_dir,
	previous_result,
	base_query_expansion_node_test,
)

sample_query = ["How many members are in Newjeans?", "What is visconde structure?"]


def test_hyde():
	generator_func = get_support_modules("llama_index_llm")
	generator_params = {"llm": "mock"}
	original_hyde = hyde.__wrapped__
	result = original_hyde(sample_query, generator_func, generator_params, prompt="")
	assert len(result[0]) == 1
	assert len(result) == 2


def test_hyde_node():
	generator_dict = {"generator_module_type": "llama_index_llm", "llm": "mock"}
	result_df = hyde(
		project_dir=project_dir, previous_result=previous_result, **generator_dict
	)
	base_query_expansion_node_test(result_df)
