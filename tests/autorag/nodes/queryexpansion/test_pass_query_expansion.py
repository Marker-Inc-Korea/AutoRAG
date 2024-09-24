from autorag.nodes.queryexpansion import PassQueryExpansion
from tests.autorag.nodes.queryexpansion.test_query_expansion_base import (
	project_dir,
	previous_result,
)


def test_pass_query_expansion():
	result_df = PassQueryExpansion.run_evaluator(
		project_dir=project_dir, previous_result=previous_result
	)
	assert len(result_df) == 5
	queries = result_df["queries"].tolist()
	assert all(isinstance(query, list) for query in queries)
	assert len(queries[0]) == 1
	assert queries == list(map(lambda x: [x], previous_result["query"].tolist()))
