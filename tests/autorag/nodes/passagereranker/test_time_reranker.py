import pytest

from autorag.nodes.passagereranker import TimeReranker

from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	contents_example,
	scores_example,
	ids_example,
	base_reranker_test,
	previous_result,
	base_reranker_node_test,
	time_list,
	project_dir_with_corpus,
)


@pytest.fixture
def time_reranker_instance(project_dir_with_corpus):
	return TimeReranker(project_dir_with_corpus)


def test_time_reranker(time_reranker_instance):
	top_k = 2

	contents_result, id_result, score_result = time_reranker_instance._pure(
		contents_example, scores_example, ids_example, top_k, time_list
	)
	assert id_result[0] == [ids_example[0][1], ids_example[0][2]]
	assert id_result[1] == [ids_example[1][1], ids_example[1][0]]
	assert contents_result[0] == [contents_example[0][1], contents_example[0][2]]
	assert contents_result[1] == [contents_example[1][1], contents_example[1][0]]
	assert score_result[0] == [0.8, 0.1]
	assert score_result[1] == [0.2, 0.1]
	base_reranker_test(contents_result, id_result, score_result, top_k)


def test_time_reranker_node(project_dir_with_corpus):
	top_k = 1
	result_df = TimeReranker.run_evaluator(
		project_dir=project_dir_with_corpus,
		previous_result=previous_result,
		top_k=top_k,
	)
	base_reranker_node_test(result_df, top_k)
