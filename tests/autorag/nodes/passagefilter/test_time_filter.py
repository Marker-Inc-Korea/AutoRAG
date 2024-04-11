from autorag.nodes.passagefilter import time_filter

from tests.autorag.nodes.passagefilter.test_passage_filter_base import queries_example, contents_example, time_list, \
    ids_example, scores_example, base_passage_filter_test, base_passage_filter_node_test, project_dir_with_corpus, \
    previous_result


def test_time_filter():
    original_time_filter = time_filter.__wrapped__
    contents_result, id_result, score_result = original_time_filter \
        (queries_example, contents_example, scores_example, ids_example, time_list, threshold="2021-06-30")
    assert id_result[0] == [ids_example[0][3]]
    assert id_result[1] == [ids_example[1][1], ids_example[1][2]]
    assert contents_result[0] == [contents_example[0][3]]
    assert contents_result[1] == [contents_example[1][1], contents_example[1][2]]
    assert score_result[0] == [0.5]
    assert score_result[1] == [0.2, 0.7]
    base_passage_filter_test(contents_result, id_result, score_result)


def test_time_filter_node(project_dir_with_corpus):
    result_df = time_filter(
        project_dir=project_dir_with_corpus, previous_result=previous_result, threshold="2021-06-30")
    base_passage_filter_node_test(result_df)
