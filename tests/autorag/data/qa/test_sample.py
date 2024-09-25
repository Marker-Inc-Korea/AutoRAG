import pandas as pd

from autorag.data.qa.sample import random_single_hop, range_single_hop
from autorag.data.qa.schema import Corpus, QA

corpus_df = pd.DataFrame(
	{"doc_id": [1, 2, 3, 4, 5], "contents": ["doc1", "doc2", "doc3", "doc4", "doc5"]}
)
corpus = Corpus(corpus_df)


def test_random_single_hop():
	qa = corpus.sample(random_single_hop, n=3)
	assert isinstance(qa, QA)
	assert len(qa.data) == 3
	assert set(qa.data.columns) == {"qid", "retrieval_gt"}
	assert qa.data["qid"].iloc[0]
	assert isinstance(qa.data["qid"].iloc[0], str)
	assert isinstance(qa.data["retrieval_gt"].iloc[0], list)
	assert qa.data["retrieval_gt"].iloc[0][0][0] in corpus_df["doc_id"].tolist()


def test_range_single_hop():
	qa = corpus.sample(range_single_hop, idx_range=range(3))
	assert isinstance(qa, QA)
	assert len(qa.data) == 3
	assert set(qa.data.columns) == {"qid", "retrieval_gt"}
	assert qa.data["qid"].iloc[0]
	assert isinstance(qa.data["qid"].iloc[0], str)
	assert isinstance(qa.data["retrieval_gt"].iloc[0], list)
	assert qa.data["retrieval_gt"].iloc[0][0][0] == 1
