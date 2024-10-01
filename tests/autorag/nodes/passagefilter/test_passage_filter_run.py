import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.nodes.passagefilter import SimilarityThresholdCutoff
from autorag.nodes.passagefilter.run import run_passage_filter_node
from autorag.utils.util import load_summary_file
from tests.autorag.nodes.passagefilter.test_passage_filter_base import (
	qa_data,
	corpus_data,
	previous_result,
)
from tests.mock import mock_get_text_embedding_batch


@pytest.fixture
def node_line_dir():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		data_dir = os.path.join(project_dir, "data")
		os.makedirs(data_dir)
		qa_data.to_parquet(os.path.join(data_dir, "qa.parquet"), index=False)
		corpus_data.to_parquet(os.path.join(data_dir, "corpus.parquet"), index=False)
		trial_dir = os.path.join(project_dir, "0")
		os.makedirs(trial_dir)
		node_line_dir = os.path.join(trial_dir, "node_line_1")
		os.makedirs(node_line_dir)
		yield node_line_dir


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_run_passage_filter_node(node_line_dir):
	modules = [SimilarityThresholdCutoff]
	module_params = [{"threshold": 0.87}]
	strategies = {
		"metrics": ["retrieval_f1", "retrieval_recall"],
		"strategy": "rank",
	}
	best_result = run_passage_filter_node(
		modules, module_params, previous_result, node_line_dir, strategies
	)
	assert os.path.exists(os.path.join(node_line_dir, "passage_filter"))
	assert set(best_result.columns) == {
		"qid",
		"query",
		"retrieval_gt",
		"generation_gt",
		"retrieved_contents",
		"retrieved_ids",
		"retrieve_scores",
		"retrieval_f1",
		"retrieval_recall",
		"passage_filter_retrieval_f1",
		"passage_filter_retrieval_recall",
	}
	# test summary feature
	summary_path = os.path.join(node_line_dir, "passage_filter", "summary.csv")
	assert os.path.exists(summary_path)
	result_path = os.path.join(node_line_dir, "passage_filter", "0.parquet")
	assert os.path.exists(result_path)
	result_df = pd.read_parquet(result_path)
	summary_df = load_summary_file(summary_path)
	assert set(summary_df.columns) == {
		"filename",
		"passage_filter_retrieval_f1",
		"passage_filter_retrieval_recall",
		"module_name",
		"module_params",
		"execution_time",
		"is_best",
	}
	assert len(summary_df) == 1
	assert summary_df["filename"][0] == "0.parquet"
	assert (
		summary_df["passage_filter_retrieval_f1"][0] == result_df["retrieval_f1"].mean()
	)
	assert (
		summary_df["passage_filter_retrieval_recall"][0]
		== result_df["retrieval_recall"].mean()
	)
	assert summary_df["module_name"][0] == "SimilarityThresholdCutoff"
	assert summary_df["module_params"][0] == {"threshold": 0.87}
	assert summary_df["execution_time"][0] > 0
	# test the best file is saved properly
	best_path = summary_df[summary_df["is_best"]]["filename"].values[0]
	assert os.path.exists(
		os.path.join(node_line_dir, "passage_filter", f"best_{best_path}")
	)
