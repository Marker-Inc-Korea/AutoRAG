from autorag.nodes.queryexpansion import hyde

sample_query = ["How many members are in Newjeans?"]


def test_hyde():
    result = hyde(sample_query)
    assert len(result[0]) == 1
