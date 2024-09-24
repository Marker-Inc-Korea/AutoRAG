import numpy as np
import pytest

from autorag.nodes.passagefilter import ThresholdCutoff
from tests.autorag.nodes.passagefilter.test_passage_filter_base import (
	contents_example,
	scores_example,
	ids_example,
	base_passage_filter_test,
	project_dir,
	previous_result,
	base_passage_filter_node_test,
)


@pytest.fixture
def threshold_cutoff_instance():
	return ThresholdCutoff(
		project_dir=project_dir, previous_result=previous_result, threshold=0.9
	)


def test_threshold_cutoff(threshold_cutoff_instance):
	contents, ids, scores = threshold_cutoff_instance._pure(
		contents_example, scores_example, ids_example, threshold=0.6
	)
	base_passage_filter_test(contents, ids, scores)


def test_threshold_cutoff_reverse(threshold_cutoff_instance):
	contents, ids, scores = threshold_cutoff_instance._pure(
		contents_example,
		scores_example,
		ids_example,
		threshold=0.4,
		reverse=True,
	)
	base_passage_filter_test(contents, ids, scores)
	assert scores[0] == [0.1, 0.1]
	assert contents[0] == ["NomaDamas is Great Team", "havertz is suck at soccer"]


def test_threshold_cutoff_numpy(threshold_cutoff_instance):
	numpy_scores = np.array([[0.1, 0.8, 0.1, 0.5], [0.1, 0.2, 0.7, 0.3]])
	contents, ids, scores = threshold_cutoff_instance._pure(
		contents_example, numpy_scores, ids_example, threshold=0.6
	)
	base_passage_filter_test(contents, ids, scores)
	assert scores == [[0.8], [0.7]]
	assert contents == [
		["Paris is the capital of France."],
		["Newjeans has 5 members."],
	]


def test_threshold_cutoff_node():
	result_df = ThresholdCutoff.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, threshold=0.9
	)
	base_passage_filter_node_test(result_df)
