import os.path
import pathlib
import shutil
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from llama_index.core import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

import autorag
from autorag.embedding.base import embedding_models
from autorag.nodes.lexicalretrieval.bm25 import BM25
from autorag.nodes.lexicalretrieval.run import run_lexical_retrieval_node
from autorag.nodes.semanticretrieval.vectordb import vectordb_ingest_api
from autorag.utils.util import load_summary_file, get_event_loop
from autorag.vectordb.chroma import Chroma
from tests.mock import mock_get_text_embedding_batch

root_dir = pathlib.PurePath(
    os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
resources_dir = os.path.join(root_dir, "resources")


@pytest.fixture
def node_line_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as test_project_dir:
        sample_project_dir = os.path.join(resources_dir, "sample_project")
        # copy & paste all folders and files in the sample_project folder
        shutil.copytree(sample_project_dir, test_project_dir, dirs_exist_ok=True)

        chroma_path = os.path.join(test_project_dir, "resources", "chroma")
        os.makedirs(chroma_path)
        corpus_path = os.path.join(test_project_dir, "data", "corpus.parquet")
        corpus_df = pd.read_parquet(corpus_path)
        embedding_models["mock_1536"] = autorag.LazyInit(MockEmbedding, embed_dim=1536)
        chroma_config = {
            "client_type": "persistent",
            "embedding_model": "mock_1536",
            "collection_name": "openai",
            "path": chroma_path,
            "similarity_metric": "cosine",
        }
        chroma = Chroma(**chroma_config)
        loop = get_event_loop()
        loop.run_until_complete(vectordb_ingest_api(chroma, corpus_df))

        chroma_config_path = os.path.join(
            test_project_dir, "resources", "vectordb.yaml"
        )
        with open(chroma_config_path, "w") as f:
            yaml.safe_dump(
                {
                    "vectordb": [
                        {**chroma_config, "name": "test_mock", "db_type": "chroma"}
                    ]
                },
                f,
            )

        test_trial_dir = os.path.join(test_project_dir, "test_trial")
        os.makedirs(test_trial_dir)
        node_line_dir = os.path.join(test_trial_dir, "test_node_line")
        os.makedirs(node_line_dir)
        yield node_line_dir


@patch.object(
    OpenAIEmbedding,
    "get_text_embedding_batch",
    mock_get_text_embedding_batch,
)
def test_run_lexical_retrieval_node(node_line_dir):
    modules = [BM25]
    module_params = [
        {"top_k": 4, "bm25_tokenizer": "gpt2"},
    ]
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    qa_path = os.path.join(project_dir, "data", "qa.parquet")
    strategies = {
        "metrics": ["retrieval_f1", "retrieval_recall"],
        "strategy": "normalize_mean",
        "speed_threshold": 5,
    }
    previous_result = pd.read_parquet(qa_path)
    best_result = run_lexical_retrieval_node(
        modules, module_params, previous_result, node_line_dir, strategies
    )
    assert os.path.exists(os.path.join(node_line_dir, "lexical_retrieval"))
    expect_columns = [
        "qid",
        "query",
        "retrieval_gt",
        "generation_gt",
        "retrieved_contents_lexical",
        "retrieved_ids_lexical",
        "retrieve_scores_lexical",
        "retrieval_f1",
        "retrieval_recall",
    ]
    assert all(
        [expect_column in best_result.columns for expect_column in expect_columns]
    )
    # test summary feature
    summary_path = os.path.join(node_line_dir, "lexical_retrieval", "summary.csv")
    bm25_top_k_path = os.path.join(node_line_dir, "lexical_retrieval", "0.parquet")
    assert os.path.exists(bm25_top_k_path)
    bm25_top_k_df = pd.read_parquet(bm25_top_k_path)
    assert os.path.exists(summary_path)
    summary_df = load_summary_file(summary_path)
    assert set(summary_df.columns) == {
        "filename",
        "retrieval_f1",
        "retrieval_recall",
        "module_name",
        "module_params",
        "execution_time",
        "is_best",
    }
    assert len(summary_df) == 1
    assert summary_df["filename"][0] == "0.parquet"
    assert summary_df["retrieval_f1"][0] == bm25_top_k_df["retrieval_f1"].mean()
    assert summary_df["retrieval_recall"][0] == bm25_top_k_df["retrieval_recall"].mean()
    assert summary_df["module_name"][0] == "BM25"
    assert summary_df["module_params"][0] == {"top_k": 4, "bm25_tokenizer": "gpt2"}
    assert summary_df["execution_time"][0] > 0

    assert summary_df["filename"].nunique() == len(summary_df)
    assert len(summary_df[summary_df["is_best"]]) == 1

    # test the best file is saved properly
    best_filename = summary_df[summary_df["is_best"]]["filename"].values[0]
    best_path = os.path.join(
        node_line_dir, "lexical_retrieval", f"best_{best_filename}"
    )
    assert os.path.exists(best_path)
    best_df = pd.read_parquet(best_path)
    assert all([expect_column in best_df.columns for expect_column in expect_columns])
