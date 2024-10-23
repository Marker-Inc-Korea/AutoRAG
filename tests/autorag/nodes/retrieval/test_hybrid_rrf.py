from unittest.mock import patch

import pandas as pd
import pytest
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.nodes.retrieval import HybridRRF
from autorag.nodes.retrieval.hybrid_rrf import rrf_pure, hybrid_rrf
from tests.autorag.nodes.retrieval.test_hybrid_base import (
	sample_ids,
	sample_scores,
	previous_result,
	pseudo_project_dir,
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


def test_hybrid_rrf_node(pseudo_project_dir):
	modules = {
		"ids": (
			[
				["id-1", "id-2", "id-3"],
				["id-1", "id-2", "id-3"],
				["id-1", "id-2", "id-3"],
			],
			[
				["id-7", "id-8", "id-9"],
				["id-7", "id-8", "id-9"],
				["id-7", "id-8", "id-9"],
			],
		),
		"scores": (
			[[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
			[[0.5, 0.6, 0.7], [0.5, 0.6, 0.7], [0.5, 0.6, 0.7]],
		),
		"top_k": 3,
		"weight": 1,
	}
	result_df = HybridRRF.run_evaluator(
		project_dir=pseudo_project_dir, previous_result=previous_result, **modules
	)
	assert len(result_df) == 3
	assert isinstance(result_df, pd.DataFrame)
	assert set(result_df.columns) == {
		"retrieved_contents",
		"retrieved_ids",
		"retrieve_scores",
	}
	assert set(result_df["retrieved_ids"].tolist()[0]) == {"id-9", "id-3", "id-2"}
	assert result_df["retrieve_scores"].tolist()[0] == pytest.approx([0.5, 0.5, 1 / 3])
	assert set(result_df["retrieved_contents"].tolist()[0]) == {
		"doc-9",
		"doc-3",
		"doc-2",
	}


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_hybrid_rrf_node_deploy(pseudo_project_dir):
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
