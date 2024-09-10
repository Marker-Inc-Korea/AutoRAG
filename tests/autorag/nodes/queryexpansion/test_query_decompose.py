from autorag.nodes.queryexpansion import query_decompose
from autorag.support import get_support_modules
from tests.autorag.nodes.queryexpansion.test_query_expansion_base import (
	project_dir,
	previous_result,
	base_query_expansion_node_test,
)

sample_query = [
	"Which group has more members, Newjeans or Aespa?",
	"Which group has more members, STAYC or Aespa?",
]


def test_query_decompose():
	generator_func = get_support_modules("llama_index_llm")
	generator_params = {"llm": "mock"}
	original_query_decompose = query_decompose.__wrapped__
	result = original_query_decompose(
		sample_query, generator_func, generator_params, prompt=""
	)
	assert len(result[0]) > 1
	assert len(result) == 2
	assert isinstance(result[0][0], str)


def test_query_decompose_node():
	generator_dict = {"generator_module_type": "llama_index_llm", "llm": "mock"}
	result_df = query_decompose(
		project_dir=project_dir, previous_result=previous_result, **generator_dict
	)
	base_query_expansion_node_test(result_df)
