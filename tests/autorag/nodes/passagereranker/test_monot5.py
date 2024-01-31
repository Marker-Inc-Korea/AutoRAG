from autorag.nodes.passagereranker import monot5

from tests.autorag.nodes.passagereranker.test_passage_reranker_base \
    import (base_reranker_test, base_reranker_node_test, queries_example,
            contents_example, scores_example, ids_example, project_dir, previous_result)


def test_monot5():
    top_k = 1
    original_monot5 = monot5.__wrapped__
    contents_result, id_result, score_result \
        = original_monot5(queries_example, contents_example, scores_example, ids_example, top_k)
    base_reranker_test(contents_result, id_result, score_result, top_k)


def test_monot5_node():
    top_k = 1
    result_df = monot5(project_dir=project_dir, previous_result=previous_result, top_k=top_k)
    base_reranker_node_test(result_df, top_k)
