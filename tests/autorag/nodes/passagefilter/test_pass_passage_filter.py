from autorag.nodes.passagefilter import PassPassageFilter
from tests.autorag.nodes.passagefilter.test_passage_filter_base import (
	project_dir,
	previous_result,
	contents_example,
	ids_example,
	scores_example,
)


def test_pass_passage_filter():
	result_df = PassPassageFilter.run_evaluator(
		project_dir=project_dir,
		previous_result=previous_result,
	)
	assert all(
		[
			column_name in result_df.columns
			for column_name in [
				"retrieved_contents",
				"retrieved_ids",
				"retrieve_scores",
			]
		]
	)
	assert result_df["retrieved_contents"].tolist() == contents_example
	assert result_df["retrieved_ids"].tolist() == ids_example
	assert result_df["retrieve_scores"].tolist() == scores_example
