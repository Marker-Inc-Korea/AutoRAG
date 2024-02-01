from tests.autorag.nodes.passagereranker.test_passage_reranker_base import base_reranker_test, queries_example, \
    contents_example, scores_example, ids_example, base_reranker_node_test, project_dir, previous_result
from autorag.nodes.passagereranker import tart


def test_tart():
    top_k = 1
    original_tart = tart.__wrapped__
    contents_result, id_result, score_result \
        = original_tart(queries_example, contents_example, scores_example, ids_example, top_k)
    base_reranker_test(contents_result, id_result, score_result, top_k)


def test_tart_node():
    top_k = 1
    result_df = tart(project_dir=project_dir, previous_result=previous_result, top_k=top_k)
    base_reranker_node_test(result_df, top_k)
