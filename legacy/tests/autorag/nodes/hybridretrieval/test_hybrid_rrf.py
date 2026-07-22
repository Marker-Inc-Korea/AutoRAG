from unittest.mock import patch

import pandas as pd
import pytest
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.nodes.hybridretrieval import HybridRRF
from autorag.nodes.hybridretrieval.hybrid_rrf import rrf_pure, hybrid_rrf
from autorag.schema.metricinput import MetricInput
from tests.autorag.nodes.retrieval.test_hybrid_base import (
    sample_ids,
    sample_scores,
    previous_result,
    pseudo_project_dir,  # noqa: F401
)
from tests.mock import mock_get_text_embedding_batch


def test_hybrid_rrf():
    result_id, result_scores = hybrid_rrf(sample_ids, sample_scores, top_k=3, weight=1)
    assert result_id == [["id-3", "id-1", "id-2"], ["id-4", "id-2", "id-3"]]
    assert result_scores[0] == pytest.approx([1.0, (1 / 4) + (1 / 3), (1 / 3)])
    assert result_scores[1] == pytest.approx([1.0, (1 / 4) + (1 / 3), (1 / 3)])


def test_rrf_pure():
    sample_ids = (["id-1", "id-2", "id-3"], ["id-1", "id-4", "id-3"])
    sample_scores = ([1, 3, 5], [4, 2, 6])
    result_id, result_scores = rrf_pure(sample_ids, sample_scores, rrf_k=1, top_k=3)
    assert result_scores == pytest.approx([1.0, (1 / 4) + (1 / 3), (1 / 3)])
    assert result_id == ["id-3", "id-1", "id-2"]


def test_hybrid_rrf_node(pseudo_project_dir):  # noqa: F811
    modules = {
        "top_k": 3,
        "strategy": {
            "metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"],
        },
        "input_metrics": [
            MetricInput(retrieval_gt=[["id-1"]]),
            MetricInput(retrieval_gt=[["id-2"]]),
        ],
    }
    result_df = HybridRRF.run_evaluator(
        project_dir=pseudo_project_dir, previous_result=previous_result, **modules
    )
    assert len(result_df["best_result"]) == 3
    assert isinstance(result_df["best_result"], pd.DataFrame)
    assert set(result_df["best_result"].columns) == {
        "retrieval_f1",
        "retrieval_recall",
        "retrieval_precision",
        "retrieved_contents",
        "retrieved_ids",
        "retrieve_scores",
    }
    assert set(result_df["best_result"]["retrieved_ids"].tolist()[0]) == {
        "id-3",
        "id-2",
        "id-1",
    }
    assert result_df["best_result"]["retrieve_scores"].tolist()[0] == pytest.approx(
        [1 / 3, 0.285714285714, 0.2]
    )
    assert set(result_df["best_result"]["retrieved_contents"].tolist()[0]) == {
        "doc-3",
        "doc-2",
        "doc-1",
    }


@patch.object(
    OpenAIEmbedding,
    "get_text_embedding_batch",
    mock_get_text_embedding_batch,
)
def test_hybrid_rrf_node_deploy(pseudo_project_dir):  # noqa: F811
    modules = {
        "target_modules": ("bm25", "vectordb"),
        "target_module_params": [
            {"top_k": 3},
            {"vectordb": "test_default", "top_k": 3},
        ],
        "top_k": 3,
        "weight": 1,
    }
    hybrid_rrf = HybridRRF(project_dir=pseudo_project_dir, **modules)
    result_df = hybrid_rrf.pure(previous_result=previous_result, **modules)
    assert len(result_df) == 3
    assert isinstance(result_df, pd.DataFrame)
    assert set(result_df.columns) == {
        "retrieved_contents",
        "retrieved_ids",
        "retrieve_scores",
    }
    assert len(result_df["retrieved_ids"].tolist()[0]) == 3
    assert len(result_df["retrieve_scores"].tolist()[0]) == 3
    assert len(result_df["retrieved_contents"].tolist()[0]) == 3
