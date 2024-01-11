from uuid import uuid4

from raground.nodes.retrieval import evenly_distribute_passages


def test_evenly_distribute_passages():
    ids = [[uuid4() for _ in range(10)] for _ in range(3)]
    scores = [[i for i in range(10)] for _ in range(3)]
    top_k = 10
    new_ids, new_scores = evenly_distribute_passages(ids, scores, top_k)
    assert len(new_ids) == top_k
    assert len(new_scores) == top_k
    assert new_scores == [0, 1, 2, 3, 0, 1, 2, 0, 1, 2]
