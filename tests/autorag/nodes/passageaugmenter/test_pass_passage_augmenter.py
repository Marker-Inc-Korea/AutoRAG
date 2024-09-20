from autorag.nodes.passageaugmenter import PassPassageAugmenter

from tests.autorag.nodes.passageaugmenter.test_base_passage_augmenter import (
	project_dir,
	previous_result,
)


def test_pass_passage_augmenter():
	result_df = PassPassageAugmenter.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, top_k=2
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
	assert (
		result_df["retrieved_contents"].tolist()
		== previous_result["retrieved_contents"].tolist()
	)
	assert (
		result_df["retrieved_ids"].tolist() == previous_result["retrieved_ids"].tolist()
	)
	assert (
		result_df["retrieve_scores"].tolist()
		== previous_result["retrieve_scores"].tolist()
	)
