import os
import pathlib
import re
import tempfile
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from llama_index.core.base.llms.types import CompletionResponse

from autorag.data.qacreation import make_single_content_qa, generate_qa_llama_index
from autorag.utils import validate_qa_dataset
from tests.mock import MockLLM

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
resource_dir = os.path.join(root_dir, "resources")


@pytest.fixture
def qa_parquet_filepath():
    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        yield f.name


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
    assert len(qa_df) == 6
    assert qa_df['retrieval_gt'].tolist()[0] == qa_df['retrieval_gt'].tolist()[1]

    assert all([len(x) == 1 and len(x[0]) == 1 for x in qa_df['retrieval_gt'].tolist()])
    assert all([len(x) == 1 for x in qa_df['generation_gt'].tolist()])
