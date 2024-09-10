from autorag.nodes.queryexpansion import multi_query_expansion
from autorag.support import get_support_modules
from tests.autorag.nodes.queryexpansion.test_query_expansion_base import (
	project_dir,
	previous_result,
	base_query_expansion_node_test,
)

sample_query = [
	"What are the potential benefits and drawbacks of implementing artificial intelligence in healthcare?",
	"How many members are in the group Aespa?",
]


def test_multi_query_expansion():
	generator_func = get_support_modules("llama_index_llm")
	generator_params = {"llm": "mock"}
	original_multi_query_expansion = multi_query_expansion.__wrapped__
	result = original_multi_query_expansion(
		sample_query, generator_func, generator_params, prompt=""
	)
	assert len(result[0]) > 1
	assert len(result) == 2
	assert isinstance(result[0][0], str)


def test_multi_query_expansion_node():
	generator_dict = {"generator_module_type": "llama_index_llm", "llm": "mock"}
	result_df = multi_query_expansion(
		project_dir=project_dir, previous_result=previous_result, **generator_dict
	)
	base_query_expansion_node_test(result_df)
