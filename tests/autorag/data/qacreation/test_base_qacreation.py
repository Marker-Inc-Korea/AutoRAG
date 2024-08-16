import os
import pathlib
import re
import tempfile
from typing import Any
from unittest.mock import patch

import chromadb
import pandas as pd
import pytest
from llama_index.core.base.llms.types import CompletionResponse

from autorag.data.qacreation import make_single_content_qa, generate_qa_llama_index, make_qa_with_existing_queries, \
    generate_answers
from autorag.utils import validate_qa_dataset
from tests.mock import MockLLM

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
resource_dir = os.path.join(root_dir, "resources")


@pytest.fixture
def qa_parquet_filepath():
    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        yield f.name


@pytest.fixture
def chroma_persistent_client():
    with tempfile.TemporaryDirectory() as temp_dir:
        client = chromadb.PersistentClient(temp_dir)
        yield client

async def acomplete_qa_creation(self, messages, **kwargs: Any):
    pattern = r'Output with (\d+) QnAs:'
    matches = re.findall(pattern, messages)
    num_questions = int(matches[-1])
    return CompletionResponse(text=
                              "[Q]: Is this the test question?\n[A]: Yes, this is the test answer." * num_questions)


@patch.object(
    MockLLM,
    "acomplete",
    acomplete_qa_creation,
)
def test_single_content_qa(qa_parquet_filepath):
    corpus_df = pd.read_parquet(os.path.join(resource_dir, "corpus_data_sample.parquet"))
    qa_df = make_single_content_qa(
        corpus_df,
        content_size=3,
        qa_creation_func=generate_qa_llama_index,
        output_filepath=qa_parquet_filepath,
        llm=MockLLM(),
        question_num_per_content=2,
        upsert=True,
    )
    validate_qa_dataset(qa_df)
    assert len(qa_df) == qa_df['qid'].nunique()
    assert len(qa_df) == 6
    assert qa_df['retrieval_gt'].tolist()[0] == qa_df['retrieval_gt'].tolist()[1]

    assert all([len(x) == 1 and len(x[0]) == 1 for x in qa_df['retrieval_gt'].tolist()])
    assert all([len(x) == 1 for x in qa_df['generation_gt'].tolist()])


@patch.object(
    MockLLM,
    "acomplete",
    acomplete_qa_creation,
)
def test_single_content_qa_long_cache_batch(qa_parquet_filepath):
    corpus_df = pd.read_parquet(os.path.join(resource_dir, "corpus_data_sample.parquet"))
    qa_df = make_single_content_qa(
        corpus_df,
        content_size=30,
        qa_creation_func=generate_qa_llama_index,
        output_filepath=qa_parquet_filepath,
        llm=MockLLM(),
        question_num_per_content=2,
        upsert=True,
        cache_batch=2,
    )
    validate_qa_dataset(qa_df)
    assert len(qa_df) == 60
    assert qa_df['retrieval_gt'].tolist()[0] == qa_df['retrieval_gt'].tolist()[1]

    assert all([len(x) == 1 and len(x[0]) == 1 for x in qa_df['retrieval_gt'].tolist()])
    assert all([len(x) == 1 for x in qa_df['generation_gt'].tolist()])


def test_make_qa_with_existing_queries(qa_parquet_filepath):
    corpus_df = pd.read_parquet(os.path.join(resource_dir, "corpus_data_sample.parquet"), engine='pyarrow')
    query_df = pd.read_parquet(os.path.join(resource_dir, "qa_data_sample.parquet"), engine='pyarrow')
    qa_df = make_qa_with_existing_queries(
        corpus_df, query_df, content_size=5, answer_creation_func=generate_answers,
        output_filepath=qa_parquet_filepath, llm=MockLLM(), upsert=True,
    )
    validate_qa_dataset(qa_df)
    assert len(qa_df) == 5
    assert all(len(elem) == 3 for elem in qa_df['retrieval_gt'].apply(lambda x: x[0]).tolist())
    assert all((elem in query_df['query'].tolist()) for elem in qa_df['query'].tolist())


def test_make_qa_with_existing_queries_persistent_client(chroma_persistent_client, qa_parquet_filepath):
    corpus_df = pd.read_parquet(os.path.join(resource_dir, "corpus_data_sample.parquet"), engine='pyarrow')
    query_df = pd.read_parquet(os.path.join(resource_dir, "qa_data_sample.parquet"), engine='pyarrow')
    collection = chroma_persistent_client.get_or_create_collection('auto-rag')
    qa_df = make_qa_with_existing_queries(
        corpus_df, query_df, content_size=5, answer_creation_func=generate_answers,
        output_filepath=qa_parquet_filepath, llm=MockLLM(), upsert=True,
        collection=collection, embedding_model='openai_embed_3_small',
    )
    validate_qa_dataset(qa_df)
    assert len(qa_df) == 5
    assert all(len(elem) == 3 for elem in qa_df['retrieval_gt'].apply(lambda x: x[0]).tolist())
    assert all((elem in query_df['query'].tolist()) for elem in qa_df['query'].tolist())
