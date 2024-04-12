from autorag.nodes.passagereranker import time_reranker

from tests.autorag.nodes.passagereranker.test_passage_reranker_base import contents_example, \
    scores_example, ids_example, base_reranker_test, previous_result, base_reranker_node_test, time_list, \
    project_dir_with_corpus


def test_time_reranker():
    top_k = 2

    original_time_reranker = time_reranker.__wrapped__
    contents_result, id_result, score_result \
        = original_time_reranker(contents_example, scores_example, ids_example, top_k, time_list)
    assert id_result[0] == [ids_example[0][1], ids_example[0][2]]
    assert id_result[1] == [ids_example[1][2], ids_example[1][1]]
    assert contents_result[0] == [contents_example[0][1], contents_example[0][2]]
    assert contents_result[1] == [contents_example[1][2], contents_example[1][1]]
    assert score_result[0] == [0.8, 0.1]
    assert score_result[1] == [0.7, 0.2]
    base_reranker_test(contents_result, id_result, score_result, top_k)


def test_time_reranker_node(project_dir_with_corpus):
    top_k = 1
    result_df = time_reranker(project_dir=project_dir_with_corpus, previous_result=previous_result, top_k=top_k)
    base_reranker_node_test(result_df, top_k)
