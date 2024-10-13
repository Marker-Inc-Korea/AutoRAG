from datetime import datetime, date

import pytest

from autorag.nodes.passagefilter import RecencyFilter
from tests.autorag.nodes.passagefilter.test_passage_filter_base import (
	contents_example,
	time_list,
	ids_example,
	scores_example,
	base_passage_filter_test,
	base_passage_filter_node_test,
	previous_result,
	project_dir_with_corpus,
)


@pytest.fixture
def recency_filter_instance(project_dir_with_corpus):
	return RecencyFilter(project_dir=project_dir_with_corpus)


def test_recency_filter(recency_filter_instance):
	contents_result, id_result, score_result = recency_filter_instance._pure(
		contents_example,
		scores_example,
		ids_example,
		time_list,
		threshold_datetime=date(2021, 6, 30),
	)
	assert id_result[0] == [ids_example[0][1], ids_example[0][2], ids_example[0][3]]
	assert id_result[1] == [ids_example[1][2], ids_example[1][3]]
	assert contents_result[0] == [
		contents_example[0][1],
		contents_example[0][2],
		contents_example[0][3],
	]
	assert contents_result[1] == [contents_example[1][2], contents_example[1][3]]
	assert score_result[0] == [0.8, 0.1, 0.5]
	assert score_result[1] == [0.7, 0.3]
	base_passage_filter_test(contents_result, id_result, score_result)


def test_recency_filter_date_time_list(recency_filter_instance):
	time_list_2 = [
		[
			date(2015, 1, 1),
			datetime(2021, 9, 3),
			datetime(2022, 3, 5, 12, 30),
			datetime(2022, 3, 5, 12, 45, 00),
		],
		[
			date(2015, 1, 1),
			date(2021, 1, 1),
			datetime(2022, 3, 5, 12, 40),
			datetime(2022, 3, 5, 12, 45, 00),
		],
	]
	contents_result, id_result, score_result = recency_filter_instance._pure(
		contents_example,
		scores_example,
		ids_example,
		time_list_2,
		threshold_datetime=date(2021, 6, 30),
	)
	assert id_result[0] == [ids_example[0][1], ids_example[0][2], ids_example[0][3]]
	assert id_result[1] == [ids_example[1][2], ids_example[1][3]]
	assert contents_result[0] == [
		contents_example[0][1],
		contents_example[0][2],
		contents_example[0][3],
	]
	assert contents_result[1] == [contents_example[1][2], contents_example[1][3]]
	assert score_result[0] == [0.8, 0.1, 0.5]
	assert score_result[1] == [0.7, 0.3]
	base_passage_filter_test(contents_result, id_result, score_result)


def test_recency_filter_all_filtered(recency_filter_instance):
	contents_result, id_result, score_result = recency_filter_instance._pure(
		contents_example,
		scores_example,
		ids_example,
		time_list,
		threshold_datetime=datetime(2040, 6, 30),
	)
	assert id_result[0] == [ids_example[0][3]]
	assert id_result[1] == [ids_example[1][3]]
	assert contents_result[0] == [contents_example[0][3]]
	assert contents_result[1] == [contents_example[1][3]]
	assert score_result[0] == [0.5]
	assert score_result[1] == [0.3]
	base_passage_filter_test(contents_result, id_result, score_result)


def test_recency_filter_minutes(recency_filter_instance):
	contents_result, id_result, score_result = recency_filter_instance._pure(
		contents_example,
		scores_example,
		ids_example,
		time_list,
		threshold_datetime=datetime(2022, 3, 5, 12, 35),
	)
	assert id_result[0] == [ids_example[0][3]]
	assert id_result[1] == [ids_example[1][2], ids_example[1][3]]
	assert contents_result[0] == [contents_example[0][3]]
	assert contents_result[1] == [contents_example[1][2], contents_example[1][3]]
	assert score_result[0] == [0.5]
	assert score_result[1] == [0.7, 0.3]
	base_passage_filter_test(contents_result, id_result, score_result)


def test_recency_filter_seconds(recency_filter_instance):
	contents_result, id_result, score_result = recency_filter_instance._pure(
		contents_example,
		scores_example,
		ids_example,
		time_list,
		threshold_datetime=datetime(2022, 3, 5, 12, 40, 30),
	)
	assert id_result[0] == [ids_example[0][3]]
	assert id_result[1] == [ids_example[1][3]]
	assert contents_result[0] == [contents_example[0][3]]
	assert contents_result[1] == [contents_example[1][3]]
	assert score_result[0] == [0.5]
	assert score_result[1] == [0.3]
	base_passage_filter_test(contents_result, id_result, score_result)


def test_recency_filter_node(project_dir_with_corpus):
	result_df = RecencyFilter.run_evaluator(
		project_dir=project_dir_with_corpus,
		previous_result=previous_result,
		threshold_datetime=datetime(2021, 6, 30),
	)
	base_passage_filter_node_test(result_df)
