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
from autorag.nodes.hybridretrieval import HybridCC, HybridRRF
from autorag.nodes.hybridretrieval.run import run_hybrid_retrieval_node
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
def test_run_hybrid_retrieval_node(node_line_dir):
    modules = [HybridCC, HybridRRF]
    module_params = [
        {"top_k": 4, "weight_range": (0.3, 0.7), "test_weight_size": 40},
        {"top_k": 4, "weight_range": (5, 70)},
    ]
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    qa_path = os.path.join(project_dir, "data", "qa.parquet")
    strategies = {
        "metrics": ["retrieval_f1", "retrieval_recall"],
        "strategy": "normalize_mean",
        "speed_threshold": 5,
    }
    previous_result = pd.read_parquet(qa_path)
    previous_result["retrieved_ids_semantic"] = [
        [
            "5b957791-3c7b-4f29-a410-8d005a538855",
            "b9ad611a-e606-4e79-885a-7f8033f42512",
            "2ec51121-3640-43d7-85db-1259cddaa4c9",
            "191c54df-703a-477d-86b6-99183f254799",
            "6f20af70-48b7-4171-a8d7-967ea583a595",
        ],
        [
            "b9ad611a-e606-4e79-885a-7f8033f42512",
            "7ac32e56-b659-43f2-a18a-d138a0973e5f",
            "191c54df-703a-477d-86b6-99183f254799",
            "6f20af70-48b7-4171-a8d7-967ea583a595",
            "dc5dc6d5-b53f-4a08-888d-2d6e5c85cf4b",
        ],
    ] * 5
    previous_result["retrieved_ids_lexical"] = [
        [
            "5b957791-3c7b-4f29-a410-8d005a538855",
            "191c54df-703a-477d-86b6-99183f254799",
            "7bdb7de8-6352-43e5-a494-d23545439df0",
            "6f20af70-48b7-4171-a8d7-967ea583a595",
            "b9ad611a-e606-4e79-885a-7f8033f42512",
        ],
        [
            "b9ad611a-e606-4e79-885a-7f8033f42512",
            "6f20af70-48b7-4171-a8d7-967ea583a595",
            "191c54df-703a-477d-86b6-99183f254799",
            "dc5dc6d5-b53f-4a08-888d-2d6e5c85cf4b",
            "2ec51121-3640-43d7-85db-1259cddaa4c9",
        ],
    ] * 5
    previous_result["retrieve_scores_semantic"] = [
        [5, 3, 1, 0.4, 0.2],
        [6, 10, 0, 1.4, 1.2],
    ] * 5
    previous_result["retrieve_scores_lexical"] = [
        [6, 10, 0, 0.5, 0.1],
        [7, 4, 2, 1.5, 1.1],
    ] * 5
    previous_result["retrieved_contents_semantic"] = [
        ["havertz", "jorginho", "mount", "sterling", "kepa"]
    ] * 10
    previous_result["retrieved_contents_lexical"] = [
        ["havertz", "jorginho", "mount", "sterling", "kepa"]
    ] * 10

    best_result = run_hybrid_retrieval_node(
        modules, module_params, previous_result, node_line_dir, strategies
    )
    assert os.path.exists(os.path.join(node_line_dir, "hybrid_retrieval"))
    expect_columns = [
        "qid",
        "query",
        "retrieval_gt",
        "generation_gt",
        "retrieved_contents",
        "retrieved_ids",
        "retrieve_scores",
        "retrieval_f1",
        "retrieval_recall",
    ]
    assert all(
        [expect_column in best_result.columns for expect_column in expect_columns]
    )
    # test summary feature
    summary_path = os.path.join(node_line_dir, "hybrid_retrieval", "summary.csv")
    vectordb_top_k_path = os.path.join(node_line_dir, "hybrid_retrieval", "0.parquet")
    assert os.path.exists(vectordb_top_k_path)
    vectordb_top_k_df = pd.read_parquet(vectordb_top_k_path)
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
    assert len(summary_df) == 2
    assert summary_df["filename"][0] == "0.parquet"
    assert summary_df["filename"][1] == "1.parquet"
    assert summary_df["retrieval_f1"][0] == vectordb_top_k_df["retrieval_f1"].mean()
    assert (
        summary_df["retrieval_recall"][0]
        == vectordb_top_k_df["retrieval_recall"].mean()
    )
    assert summary_df["module_name"][0] == "HybridCC"
    assert summary_df["module_name"][1] == "HybridRRF"
    assert "weight" in summary_df["module_params"][0].keys()
    assert "weight" in summary_df["module_params"][1].keys()
    assert summary_df["execution_time"][0] > 0

    assert summary_df["filename"].nunique() == len(summary_df)
    assert len(summary_df[summary_df["is_best"]]) == 1

    # test the best file is saved properly
    best_filename = summary_df[summary_df["is_best"]]["filename"].values[0]
    best_path = os.path.join(node_line_dir, "hybrid_retrieval", f"best_{best_filename}")
    assert os.path.exists(best_path)
    best_df = pd.read_parquet(best_path)
    assert all([expect_column in best_df.columns for expect_column in expect_columns])
