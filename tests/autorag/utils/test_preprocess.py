from datetime import datetime

import pytest
import pandas as pd

from autorag.utils import validate_qa_dataset, validate_corpus_dataset, cast_qa_dataset, cast_corpus_dataset


@pytest.fixture
def qa_df():
    return pd.DataFrame({
        'qid': ['id1', 'id2'],
        'query': ['query1', 'query2'],
        'retrieval_gt': ['answer1', 'answer2'],
        'generation_gt': 'answer1',
    })


# Fixture for corpus dataset
@pytest.fixture
def corpus_df():
    return pd.DataFrame({
        'doc_id': ['doc1', 'doc2', 'doc3'],
        'contents': ['content1', 'content2', 'content3'],
        'metadata': [{}, {'test_key': 'test_value'}, {'last_modified_datetime': datetime(2022, 12, 1, 3, 4, 5)}]
    })


# Test validate_qa_dataset
def test_validate_qa_dataset(qa_df):
    # This should pass as the qa_df fixture contains the required columns
    validate_qa_dataset(qa_df)

    # This should fail as 'qid' column is missing
    with pytest.raises(AssertionError):
        invalid_df = qa_df.drop(columns=['qid'])
        validate_qa_dataset(invalid_df)


# Test validate_corpus_dataset
def test_validate_corpus_dataset(corpus_df):
    # This should pass as the corpus_df fixture contains the required columns
    validate_corpus_dataset(corpus_df)

    # This should fail as 'doc_id' column is missing
    with pytest.raises(AssertionError):
        invalid_df = corpus_df.drop(columns=['doc_id'])
        validate_corpus_dataset(invalid_df)


# Test cast_qa_dataset
def test_cast_qa_dataset(qa_df):
    # Cast the dataset and check for correct casting
    casted_df = cast_qa_dataset(qa_df)
    assert all(isinstance(x, list) for x in casted_df['retrieval_gt'])
    assert all(isinstance(x[0], list) for x in casted_df['retrieval_gt'])
    assert all(isinstance(x, list) for x in casted_df['generation_gt'])

    # Check for ValueError when 'retrieval_gt' has an incorrect type
    with pytest.raises(ValueError):
        invalid_df = qa_df.copy()
        invalid_df.at[0, 'retrieval_gt'] = 123  # Invalid type for casting
        cast_qa_dataset(invalid_df)


# Test cast_corpus_dataset
def test_cast_corpus_dataset(corpus_df):
    # Cast the dataset and check for a datetime key in metadata
    casted_df = cast_corpus_dataset(corpus_df)
    assert all('last_modified_datetime' in x for x in casted_df['metadata'])
    assert casted_df['metadata'].iloc[1]['test_key'] == 'test_value'
    assert casted_df['metadata'].iloc[2]['last_modified_datetime'] == datetime(2022, 12, 1, 3, 4, 5)
