from raground.nodes.retrieval.base import evenly_distribute_passages

queries = [
    ["What is Visconde structure?", "What are Visconde structure?"],
    ["What is the structure of StrategyQA dataset in this paper?"],
    ["What's your favorite source of RAG framework?",
     "What is your source of RAG framework?",
     "Is RAG framework have source?"],
]


def test_evenly_distribute_passages():
    ids = [[f'test-{i}-{j}' for i in range(10)] for j in range(3)]
    scores = [[i for i in range(10)] for _ in range(3)]
    top_k = 10
    new_ids, new_scores = evenly_distribute_passages(ids, scores, top_k)
    assert len(new_ids) == top_k
    assert len(new_scores) == top_k
    assert new_scores == [0, 1, 2, 3, 0, 1, 2, 0, 1, 2]
