import test_base_reranker
from autorag.nodes.passagereranker import upr

def test_upr():
    test_base_reranker.rerank_test(upr.upr_rerank)