from autorag.support import get_support_modules, get_support_nodes


def test_get_support_modules():
    result = get_support_modules("bm25")
    assert result.__name__ == "BM25"


def test_get_support_nodes():
    result = get_support_nodes("lexical_retrieval")
    assert result.__name__ == "run_lexical_retrieval_node"
