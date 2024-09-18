import pytest
from sklearn.utils.fixes import percentile

from autorag.nodes.passagefilter import percentile_cutoff, PercentileCutoff
from tests.autorag.nodes.passagefilter.test_passage_filter_base import (
	queries_example,
	contents_example,
	scores_example,
	ids_example,
	base_passage_filter_test,
	project_dir,
	previous_result,
	base_passage_filter_node_test,
)


@pytest.fixture
def percentile_cutoff_instance():
	return PercentileCutoff(
		project_dir=project_dir, previous_result=previous_result, percentile=0.6
	)


def test_percentile_cutoff(percentile_cutoff_instance):
	contents, ids, scores = percentile_cutoff_instance._pure(
		queries_example, contents_example, scores_example, ids_example, percentile=0.6
	)
	base_passage_filter_test(contents, ids, scores)
	assert scores[0] == [0.8, 0.5]
	assert contents[0] == [
		"Paris is the capital of France.",
		"Paris is one of the capital from France. Isn't it?",
	]


def test_percentile_cutoff_reverse(percentile_cutoff_instance):
	contents, ids, scores = percentile_cutoff_instance._pure(
		queries_example,
		contents_example,
		scores_example,
		ids_example,
		percentile=0.6,
		reverse=True,
	)
	base_passage_filter_test(contents, ids, scores)
	assert scores[0] == [0.1, 0.1]
	assert contents[0] == ["NomaDamas is Great Team", "havertz is suck at soccer"]


def test_percentile_cutoff_node():
	result_df = PercentileCutoff.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, percentile=0.9
	)
	base_passage_filter_node_test(result_df)
