from autorag.nodes.queryexpansion import query_decompose


sample_query = ["Which group has more members, Newjeans or Espa?"]


def test_query_decompose():
    result = query_decompose(sample_query)
    assert len(result[0]) > 1
