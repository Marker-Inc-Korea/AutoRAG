import os
import pathlib
import pickle
from uuid import UUID

from raground.nodes.retrieval import bm25

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
bm25_path = os.path.join(root_dir, "resources", "test_bm25_retrieval.pkl")
with open(bm25_path, 'rb') as r:
    bm25_corpus = pickle.load(r)

queries = [
    ["What is Visconde structure?", "What are Visconde structure?"],
    ["What is the structure of StrategyQA dataset in this paper?"],
    ["What's your favorite source of RAG framework?",
     "What is your source of RAG framework?",
     "Is RAG framework have source?"],
]


def test_bm25_retrieval():
    top_k = 10
    original_bm25 = bm25.__wrapped__
    result = original_bm25(queries, top_k=top_k, bm25_corpus=bm25_corpus)
    assert len(result) == 3
    for each_result in result:
        assert isinstance(each_result, tuple)
        assert len(each_result) == 2
        ids = each_result[0]
        scores = each_result[1]
        assert isinstance(ids, list)
        assert isinstance(scores, list)
        assert len(ids) == top_k
        assert len(scores) == top_k
        for _id, score in zip(ids, scores):
            assert isinstance(_id, UUID)
            assert isinstance(score, float)
        for i in range(1, len(scores)):
            assert scores[i - 1] >= scores[i]
