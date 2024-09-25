import os

import pandas as pd


def validate_corpus(result_df: pd.DataFrame, length: int, parquet_filepath):
	assert isinstance(result_df, pd.DataFrame)
	assert len(result_df) == length
	assert "doc_id" in result_df.columns
	assert "contents" in result_df.columns
	assert "metadata" in result_df.columns
	assert os.path.exists(parquet_filepath)

	assert ["test text"] * length == result_df["contents"].tolist()
	assert all(
		[
			"last_modified_datetime" in metadata
			for metadata in result_df["metadata"].tolist()
		]
	)
	assert all([isinstance(doc_id, str) for doc_id in result_df["doc_id"].tolist()])
