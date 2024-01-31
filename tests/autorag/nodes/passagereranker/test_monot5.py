from autorag.nodes.passagereranker import monot5

from tests.autorag.nodes.passagereranker.test_base_reranker import (base_reranker_test, queries_example,
                                                                    contents_example, scores_example, ids_example)


def test_monot5():
    contents_result, id_result, score_result\
        = monot5(queries_example, contents_example, scores_example, ids_example)
    base_reranker_test(contents_result, id_result, score_result)
