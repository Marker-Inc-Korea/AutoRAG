import pandas as pd
import pytest
import os
import pathlib

from autorag.data.qacreation.simple import generate_simple_qa_dataset, generate_qa_row
from autorag.utils.preprocess import validate_qa_dataset
from autorag.data.utils.llamaindex import get_file_metadata, llama_documents_to_parquet

from llama_index import SimpleDirectoryReader
from guidance import models


root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent

raw_dir = os.path.join(root_dir, "resources", "data_creation", "raw_dir")

load_dir = os.path.join(root_dir, "resources", "data_creation", "corpus_dir")
load_file_name = "test_corpus.parquet"

output_filepath = os.path.join(root_dir, "resources", "data_creation", "qa_dir", "test_simple_qa_dataset.parquet")

def test_generate_simple_qa_dataset():
    loader = SimpleDirectoryReader(
        file_metadata=get_file_metadata,
        input_dir=raw_dir
    )

    documents = loader.load_data(
        show_progress=True,
        num_workers=0
    )
    llama_documents_to_parquet(llama_documents=documents,output_filepath=os.path.join(load_dir, load_file_name))

    qa_dataset = generate_simple_qa_dataset(corpus_data=pd.read_parquet(os.path.join(load_dir, load_file_name)), llm=models.OpenAI("gpt-3.5-turbo"),
                                         output_filepath=output_filepath, generate_row_function=generate_qa_row)
    validate_qa_dataset(qa_dataset)