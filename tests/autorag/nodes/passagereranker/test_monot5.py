from autorag.nodes.passagereranker import monot5

import test_base_reranker


def test_monot5():
    test_base_reranker.rerank_test(monot5)
