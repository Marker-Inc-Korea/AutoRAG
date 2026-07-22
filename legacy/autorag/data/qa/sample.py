import uuid
from typing import Iterable

import pandas as pd


def random_single_hop(
	corpus_df: pd.DataFrame, n: int, random_state: int = 42
) -> pd.DataFrame:
	sample_df = corpus_df.sample(n, random_state=random_state)
	return pd.DataFrame(
		{
			"qid": [str(uuid.uuid4()) for _ in range(len(sample_df))],
			"retrieval_gt": [[[id_]] for id_ in sample_df["doc_id"].tolist()],
		}
	)


def range_single_hop(corpus_df: pd.DataFrame, idx_range: Iterable):
	sample_df = corpus_df.iloc[idx_range]
	return pd.DataFrame(
		{
			"qid": [str(uuid.uuid4()) for _ in range(len(sample_df))],
			"retrieval_gt": [[[id_]] for id_ in sample_df["doc_id"].tolist()],
		}
	)
