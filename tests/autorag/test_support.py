from autorag.support import get_support_modules, get_support_nodes


def test_get_support_modules():
    result = get_support_modules('bm25')
    assert result.__name__ == 'bm25'


def test_get_support_nodes():
    result = get_support_nodes('retrieval')
    assert result.__name__ == 'run_retrieval_node'
