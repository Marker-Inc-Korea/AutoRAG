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
from autorag.nodes.retrieval import BM25, VectorDB, HybridCC, HybridRRF
from autorag.nodes.retrieval.run import run_retrieval_node
from autorag.nodes.retrieval.vectordb import vectordb_ingest
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
		autorag.embedding_models["mock_1536"] = autorag.LazyInit(
			MockEmbedding, embed_dim=1536
		)
		chroma_config = {
			"client_type": "persistent",
			"embedding_model": "mock_1536",
			"collection_name": "openai",
			"path": chroma_path,
			"similarity_metric": "cosine",
		}
		chroma = Chroma(**chroma_config)
		loop = get_event_loop()
		loop.run_until_complete(vectordb_ingest(chroma, corpus_df))

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
def test_run_retrieval_node(node_line_dir):
	modules = [BM25, VectorDB, HybridRRF, HybridCC, HybridCC]
	module_params = [
		{"top_k": 4, "bm25_tokenizer": "gpt2"},
		{"top_k": 4, "vectordb": "test_mock"},
		{"top_k": 4, "weight_range": (5, 70)},
		{"top_k": 4, "weight_range": (0.3, 0.7), "test_weight_size": 40},
		{"top_k": 4, "weight_range": (0.1, 0.9), "test_weight_size": 8},
	]
	project_dir = pathlib.PurePath(node_line_dir).parent.parent
	qa_path = os.path.join(project_dir, "data", "qa.parquet")
	strategies = {
		"metrics": ["retrieval_f1", "retrieval_recall"],
		"strategy": "normalize_mean",
		"speed_threshold": 5,
	}
	previous_result = pd.read_parquet(qa_path)
	best_result = run_retrieval_node(
		modules, module_params, previous_result, node_line_dir, strategies
	)
	assert os.path.exists(os.path.join(node_line_dir, "retrieval"))
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
	summary_path = os.path.join(node_line_dir, "retrieval", "summary.csv")
	bm25_top_k_path = os.path.join(node_line_dir, "retrieval", "1.parquet")
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
	assert len(summary_df) == 5
	assert summary_df["filename"][0] == "0.parquet"
	assert summary_df["retrieval_f1"][1] == bm25_top_k_df["retrieval_f1"].mean()
	assert summary_df["retrieval_recall"][1] == bm25_top_k_df["retrieval_recall"].mean()
	assert summary_df["module_name"][0] == "VectorDB"
	assert summary_df["module_params"][0] == {"top_k": 4, "vectordb": "test_mock"}
	assert summary_df["execution_time"][0] > 0
	# assert average times
	assert summary_df["execution_time"][0] + summary_df["execution_time"][
		1
	] == pytest.approx(summary_df["execution_time"][2])
	assert summary_df["execution_time"][0] + summary_df["execution_time"][
		1
	] == pytest.approx(summary_df["execution_time"][3])

	assert summary_df["filename"].nunique() == len(summary_df)
	assert len(summary_df[summary_df["is_best"]]) == 1

	# test summary_df hybrid retrieval convert well
	assert all(summary_df["module_params"].apply(lambda x: "ids" not in x))
	assert all(summary_df["module_params"].apply(lambda x: "scores" not in x))
	hybrid_summary_df = summary_df[summary_df["module_name"].str.contains("hybrid")]
	assert all(
		hybrid_summary_df["module_params"].apply(lambda x: "target_modules" in x)
	)
	assert all(
		hybrid_summary_df["module_params"].apply(lambda x: "target_module_params" in x)
	)
	assert all(
		hybrid_summary_df["module_params"].apply(
			lambda x: x["target_modules"] == ("vectordb", "bm25")
		)
	)
	assert all(
		hybrid_summary_df["module_params"].apply(
			lambda x: x["target_module_params"]
			== (
				{"top_k": 4, "embedding_model": "openai"},
				{"top_k": 4, "bm25_tokenizer": "gpt2"},
			)
		)
	)
	assert all(hybrid_summary_df["module_params"].apply(lambda x: "weight" in x))
	# test the best file is saved properly
	best_filename = summary_df[summary_df["is_best"]]["filename"].values[0]
	best_path = os.path.join(node_line_dir, "retrieval", f"best_{best_filename}")
	assert os.path.exists(best_path)
	best_df = pd.read_parquet(best_path)
	assert all([expect_column in best_df.columns for expect_column in expect_columns])


@pytest.fixture
def pseudo_node_dir():
	summary_df = pd.DataFrame(
		{
			"filename": ["0.parquet", "1.parquet", "2.parquet"],
			"module_name": ["bm25", "vectordb", "vectordb"],
			"module_params": [
				{"top_k": 3},
				{"top_k": 3, "embedding_model": "openai"},
				{"top_k": 3, "embedding_model": "huggingface"},
			],
			"execution_time": [1, 1, 1],
			"retrieval_f1": [0.1, 0.2, 0.3],
			"retrieval_recall": [0.2, 0.55, 0.5],
		}
	)
	bm25_df = pd.DataFrame(
		{
			"query": ["query-1", "query-2", "query-3"],
			"retrieved_ids": [
				["id-1", "id-2", "id-3"],
				["id-1", "id-2", "id-3"],
				["id-1", "id-2", "id-3"],
			],
			"retrieve_scores": [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
			"retrieval_f1": [0.05, 0.1, 0.15],
			"retrieval_recall": [0.1, 0.275, 0.25],
		}
	)
	vector_openai_df = pd.DataFrame(
		{
			"query": ["query-1", "query-2", "query-3"],
			"retrieved_ids": [
				["id-4", "id-5", "id-6"],
				["id-4", "id-5", "id-6"],
				["id-4", "id-5", "id-6"],
			],
			"retrieve_scores": [[0.3, 0.4, 0.5], [0.3, 0.4, 0.5], [0.3, 0.4, 0.5]],
			"retrieval_f1": [0.15, 0.2, 0.25],
			"retrieval_recall": [0.3, 0.55, 0.5],
		}
	)
	vector_huggingface_df = pd.DataFrame(
		{
			"query": ["query-1", "query-2", "query-3"],
			"retrieved_ids": [
				["id-7", "id-8", "id-9"],
				["id-7", "id-8", "id-9"],
				["id-7", "id-8", "id-9"],
			],
			"retrieve_scores": [[0.5, 0.6, 0.7], [0.5, 0.6, 0.7], [0.5, 0.6, 0.7]],
			"retrieval_f1": [0.25, 0.3, 0.35],
			"retrieval_recall": [0.5, 0.825, 0.75],
		}
	)

	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as node_dir:
		summary_df.to_csv(os.path.join(node_dir, "summary.csv"))
		bm25_df.to_parquet(os.path.join(node_dir, "0.parquet"))
		vector_openai_df.to_parquet(os.path.join(node_dir, "1.parquet"))
		vector_huggingface_df.to_parquet(os.path.join(node_dir, "2.parquet"))
		yield node_dir


@patch.object(
	OpenAIEmbedding,
	"get_text_embedding_batch",
	mock_get_text_embedding_batch,
)
def test_run_retrieval_node_only_hybrid(node_line_dir):
	modules = [HybridCC]
	module_params = [
		{
			"top_k": 4,
			"target_modules": ("bm25", "vectordb"),
			"weight": 0.3,
			"target_module_params": (
				{"top_k": 3},
				{"top_k": 3, "vectordb": "test_mock"},
			),
		},
	]
	project_dir = pathlib.PurePath(node_line_dir).parent.parent
	qa_path = os.path.join(project_dir, "data", "qa.parquet")
	strategies = {
		"metrics": ["retrieval_f1", "retrieval_recall"],
	}
	previous_result = pd.read_parquet(qa_path)
	best_result = run_retrieval_node(
		modules, module_params, previous_result, node_line_dir, strategies
	)
	assert os.path.exists(os.path.join(node_line_dir, "retrieval"))
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
	summary_path = os.path.join(node_line_dir, "retrieval", "summary.csv")
	single_result_path = os.path.join(node_line_dir, "retrieval", "0.parquet")
	assert os.path.exists(single_result_path)
	single_result_df = pd.read_parquet(single_result_path)
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
	assert summary_df["retrieval_f1"][0] == single_result_df["retrieval_f1"].mean()
	assert (
		summary_df["retrieval_recall"][0] == single_result_df["retrieval_recall"].mean()
	)
	assert summary_df["module_name"][0] == "HybridCC"
	assert summary_df["module_params"][0] == {
		"top_k": 4,
		"target_modules": ("bm25", "vectordb"),
		"weight": 0.3,
		"target_module_params": (
			{"top_k": 3},
			{"top_k": 3, "vectordb": "test_mock"},
		),
	}
	assert summary_df["execution_time"][0] > 0
	assert summary_df["is_best"][0]
	assert summary_df["filename"].nunique() == len(summary_df)

	# test the best file is saved properly
	best_filename = summary_df[summary_df["is_best"]]["filename"].values[0]
	best_path = os.path.join(node_line_dir, "retrieval", f"best_{best_filename}")
	assert os.path.exists(best_path)
	best_df = pd.read_parquet(best_path)
	assert all([expect_column in best_df.columns for expect_column in expect_columns])
