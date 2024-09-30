import os.path
import pathlib
import tempfile

import pandas as pd
import pytest

from autorag import generator_models
from autorag.nodes.passagecompressor import TreeSummarize, Refine
from autorag.nodes.passagecompressor.run import run_passage_compressor_node
from autorag.utils.util import load_summary_file
from tests.mock import MockLLM

root_dir = pathlib.PurePath(
	os.path.dirname(os.path.realpath(__file__))
).parent.parent.parent
resources_dir = os.path.join(root_dir, "resources")
qa_data = pd.DataFrame(
	{
		"qid": ["id-1", "id-2", "id-3"],
		"query": ["query-1", "query-2", "query-3"],
		"retrieval_gt": [
			[["doc-1", "doc-2"]],
			[["doc-3"]],
			[["doc-5"], ["doc-6"]],
		],
		"generation_gt": [["generation-1"], ["generation-2"], ["generation-3"]],
	}
)
corpus_data = pd.DataFrame(
	{
		"doc_id": ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5", "doc-6"],
		"contents": [
			"Enough for drinking water",
			"Just looking for a water bottle",
			"Do you want to buy some?",
			"I want to buy great octopus",
			"I want to buy some water",
			"Who is son? He is great player in the world",
		],
		"metadata": [
			{"datetime": "2021-01-01"},
			{"datetime": "2021-01-02"},
			{"datetime": "2021-01-03"},
			{"datetime": "2021-01-04"},
			{"datetime": "2021-01-05"},
			{"datetime": "2021-01-06"},
		],
	}
)
previous_result = pd.concat(
	[
		qa_data,
		pd.DataFrame(
			{
				"retrieved_ids": [
					["doc-1", "doc-3"],
					["doc-2", "doc-5"],
					["doc-5", "doc-6"],
				],
				"retrieved_contents": [
					["Enough for drinking water", "Do you want to buy some?"],
					["Just looking for a water bottle", "I want to buy some water"],
					[
						"I want to buy some water",
						"Who is son? He is great player in the world",
					],
				],
				"retrieve_scores": [
					[0.1, 0.2],
					[0.3, 0.4],
					[0.5, 0.6],
				],
			}
		),
	],
	axis=1,
)


@pytest.fixture
def node_line_dir():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		data_dir = os.path.join(project_dir, "data")
		os.makedirs(data_dir)
		qa_data.to_parquet(os.path.join(data_dir, "qa.parquet"), index=False)
		corpus_data.to_parquet(os.path.join(data_dir, "corpus.parquet"), index=False)
		trial_dir = os.path.join(project_dir, "trial_1")
		os.makedirs(trial_dir)
		node_line_dir = os.path.join(trial_dir, "node_line_1")
		os.makedirs(node_line_dir)
		yield node_line_dir


def test_run_passage_compressor_node(node_line_dir):
	generator_models["mock"] = MockLLM
	modules = [TreeSummarize, Refine]
	module_params = [
		{"llm": "mock", "model": "gpt-3.5-turbo-16k", "batch": 4},
		{"llm": "mock", "model": "gpt-3.5-turbo"},
	]
	strategies = {
		"metrics": ["retrieval_token_f1", "retrieval_token_precision"],
		"strategy": "normalize_mean",
		"speed_threshold": 5,
	}
	best_result = run_passage_compressor_node(
		modules, module_params, previous_result, node_line_dir, strategies
	)
	assert os.path.exists(os.path.join(node_line_dir, "passage_compressor"))
	assert set(best_result.columns) == {
		"qid",
		"query",
		"retrieval_gt",
		"generation_gt",
		"retrieved_contents",
		"retrieved_ids",
		"retrieve_scores",
		"passage_compressor_retrieval_token_f1",
		"passage_compressor_retrieval_token_precision",
	}
	# test summary feature
	summary_path = os.path.join(node_line_dir, "passage_compressor", "summary.csv")
	single_result_path = os.path.join(node_line_dir, "passage_compressor", "0.parquet")
	assert os.path.exists(single_result_path)
	single_result_df = pd.read_parquet(single_result_path)
	assert os.path.exists(summary_path)
	summary_df = load_summary_file(summary_path)
	assert set(summary_df.columns) == {
		"filename",
		"passage_compressor_retrieval_token_f1",
		"passage_compressor_retrieval_token_precision",
		"module_name",
		"module_params",
		"execution_time",
		"is_best",
	}
	assert len(summary_df) == 2
	assert summary_df["filename"][0] == "0.parquet"
	assert summary_df["passage_compressor_retrieval_token_f1"][0] == pytest.approx(
		single_result_df["retrieval_token_f1"].mean()
	)
	assert summary_df["passage_compressor_retrieval_token_precision"][
		0
	] == pytest.approx(single_result_df["retrieval_token_precision"].mean())
	assert summary_df["module_name"][0] == "TreeSummarize"
	assert summary_df["module_name"][1] == "Refine"
	assert summary_df["module_params"][0] == {
		"llm": "mock",
		"model": "gpt-3.5-turbo-16k",
		"batch": 4,
	}
	assert summary_df["execution_time"][0] > 0
	# test the best file is saved properly
	best_path = summary_df[summary_df["is_best"]]["filename"].values[0]
	assert os.path.exists(
		os.path.join(node_line_dir, "passage_compressor", f"best_{best_path}")
	)
