import test_base_reranker
from autorag.nodes.passagereranker import tart


def test_tart():
    test_base_reranker.rerank_test(tart)
