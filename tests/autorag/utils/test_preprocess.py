import os
import pathlib
from datetime import datetime

import pandas as pd
import pytest

from autorag.utils import (
	validate_qa_dataset,
	validate_corpus_dataset,
	cast_qa_dataset,
	cast_corpus_dataset,
	validate_qa_from_corpus_dataset,
)


@pytest.fixture
def qa_df():
	return pd.DataFrame(
		{
			"qid": ["id1", "id2", "id3"],
			"query": ["query1", "query2", "query3"],
			"retrieval_gt": [[["doc1", "doc3"], ["doc2"]], [[]], [[]]],
			"generation_gt": "answer1",
		}
	)


# Fixture for corpus dataset
@pytest.fixture
def corpus_df():
	return pd.DataFrame(
		{
			"doc_id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
			"contents": ["content1", "content2", "content3", None, " \n  \t 	"],
			"metadata": [
				{"prev_id": None, "next_id": "doc2"},
				{"test_key": "test_value"},
				{"last_modified_datetime": datetime(2022, 12, 1, 3, 4, 5)},
				{"test_key": "test_value"},
				{"test_key": "test_value"},
			],
		}
	)


# Test validate_qa_dataset
def test_validate_qa_dataset(qa_df):
	# This should pass as the qa_df fixture contains the required columns
	validate_qa_dataset(qa_df)

	# This should fail as 'qid' column is missing
	with pytest.raises(AssertionError):
		invalid_df = qa_df.drop(columns=["qid"])
		validate_qa_dataset(invalid_df)


# Test validate_corpus_dataset
def test_validate_corpus_dataset(corpus_df):
	# This should pass as the corpus_df fixture contains the required columns
	validate_corpus_dataset(corpus_df)

	# This should fail as 'doc_id' column is missing
	with pytest.raises(AssertionError):
		invalid_df = corpus_df.drop(columns=["doc_id"])
		validate_corpus_dataset(invalid_df)


# Test cast_qa_dataset
def test_cast_qa_dataset(qa_df):
	# Cast the dataset and check for correct casting
	qa_df.drop(index=1, inplace=True)
	casted_df = cast_qa_dataset(qa_df)
	assert all(isinstance(x, list) for x in casted_df["retrieval_gt"])
	assert all(isinstance(x[0], list) for x in casted_df["retrieval_gt"])
	assert all(isinstance(x, list) for x in casted_df["generation_gt"])

	# Check for ValueError when 'retrieval_gt' has an incorrect type
	with pytest.raises(ValueError):
		invalid_df = qa_df.copy()
		invalid_df.at[0, "retrieval_gt"] = 123  # Invalid type for casting
		cast_qa_dataset(invalid_df)


# Test cast_corpus_dataset
def test_cast_corpus_dataset(corpus_df):
	# Cast the dataset and check for a datetime key in metadata
	casted_df = cast_corpus_dataset(corpus_df)
	assert len(casted_df) == 3
	assert all("last_modified_datetime" in x for x in casted_df["metadata"])
	assert all(
		isinstance(x["last_modified_datetime"], datetime) for x in casted_df["metadata"]
	)
	assert casted_df["metadata"].iloc[0]["prev_id"] is None
	assert casted_df["metadata"].iloc[0]["next_id"] == "doc2"
	assert casted_df["metadata"].iloc[1]["prev_id"] is None
	assert casted_df["metadata"].iloc[1]["next_id"] is None
	assert casted_df["metadata"].iloc[2]["next_id"] is None
	assert casted_df["metadata"].iloc[2]["last_modified_datetime"] == datetime(
		2022, 12, 1, 3, 4, 5
	)


def test_validate_qa_from_corpus_dataset(qa_df, corpus_df):
	validate_qa_from_corpus_dataset(qa_df, corpus_df)

	with pytest.raises(AssertionError) as excinfo:
		invalid_df = qa_df.copy()
		invalid_df.at[0, "retrieval_gt"] = [["answer1", "answer2"], ["answer3"]]
		validate_qa_from_corpus_dataset(invalid_df, corpus_df)
	assert "3 doc_ids in retrieval_gt do not exist in corpus_df." in str(excinfo.value)

	root_dir = pathlib.PurePath(
		os.path.dirname(os.path.realpath(__file__))
	).parent.parent
	project_dir = os.path.join(root_dir, "resources", "sample_project")
	qa_parquet = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))

	with pytest.raises(AssertionError) as excinfo:
		validate_qa_from_corpus_dataset(qa_parquet, corpus_df)
	assert "10 doc_ids in retrieval_gt do not exist in corpus_df." in str(excinfo.value)
