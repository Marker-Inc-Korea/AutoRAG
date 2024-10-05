import os
import pathlib
import tempfile

import pandas as pd
import pytest
from llama_index.core import SimpleDirectoryReader

from autorag.data.legacy.corpus.llama_index import llama_documents_to_parquet
from autorag.data.legacy.qacreation.simple import (
	generate_simple_qa_dataset,
	generate_qa_row,
)
from autorag.data.utils.util import get_file_metadata
from autorag.utils.preprocess import validate_qa_dataset, validate_corpus_dataset
from tests.delete_tests import is_github_action

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent.parent

raw_dir = os.path.join(root_dir, "resources", "data_creation", "raw_dir")

load_file_name = "test_corpus.parquet"


@pytest.fixture
def load_dir():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
		yield temp_dir


@pytest.fixture
def output_filedir():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
		yield temp_dir


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it will be deprecated.",
)
def test_generate_simple_qa_dataset(load_dir, output_filedir):
	from guidance import models

	loader = SimpleDirectoryReader(file_metadata=get_file_metadata, input_dir=raw_dir)

	documents = loader.load_data(show_progress=True, num_workers=0)
	llama_documents_to_parquet(
		llama_documents=documents,
		output_filepath=os.path.join(load_dir, load_file_name),
	)
	assert os.path.exists(os.path.join(load_dir, load_file_name))
	corpus_data = pd.read_parquet(os.path.join(load_dir, load_file_name))
	validate_corpus_dataset(corpus_data)

	qa_dataset = generate_simple_qa_dataset(
		corpus_data=pd.read_parquet(os.path.join(load_dir, load_file_name)),
		llm=models.OpenAI("gpt-3.5-turbo"),
		output_filepath=os.path.join(output_filedir, "qa.parquet"),
		generate_row_function=generate_qa_row,
	)
	validate_qa_dataset(qa_dataset)
	assert os.path.exists(os.path.join(output_filedir, "qa.parquet"))
