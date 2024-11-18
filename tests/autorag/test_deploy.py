import asyncio
import logging
import os
import pathlib
import tempfile

import nest_asyncio
import pandas as pd
import pytest
import yaml

from autorag.deploy import (
	summary_df_to_yaml,
	extract_best_config,
	Runner,
	extract_node_line_names,
	extract_node_strategy,
)
from autorag.deploy.api import ApiRunner
from autorag.evaluator import Evaluator
from tests.delete_tests import is_github_action

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
resource_dir = os.path.join(root_dir, "resources")

logger = logging.getLogger("AutoRAG")


@pytest.fixture
def evaluator():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		evaluator = Evaluator(
			os.path.join(resource_dir, "qa_data_sample.parquet"),
			os.path.join(resource_dir, "corpus_data_sample.parquet"),
			project_dir=project_dir,
		)
		yield evaluator


@pytest.fixture
def evaluator_trial_done(evaluator):
	evaluator.start_trial(os.path.join(resource_dir, "simple_with_llm.yaml"))
	yield evaluator


@pytest.fixture
def full_config():
	yaml_path = os.path.join(resource_dir, "full.yaml")
	with open(yaml_path, "r") as f:
		yaml_dict = yaml.safe_load(f)
	return yaml_dict


summary_df = pd.DataFrame(
	{
		"node_line_name": ["node_line_2", "node_line_2", "node_line_1"],
		"node_type": ["retrieval", "rerank", "generation"],
		"best_module_filename": [
			"bm25=>top_k_50.parquet",
			"upr=>model_llama-2-havertz_chelsea.parquet",
			"gpt-4=>top_p_0.9.parquet",
		],
		"best_module_name": ["bm25", "upr", "gpt-4"],
		"best_module_params": [
			{"top_k": 50},
			{"model": "llama-2", "havertz": "chelsea"},
			{"top_p": 0.9},
		],
		"best_execution_time": [1.0, 0.5, 2.0],
	}
)
solution_dict = {
	"node_lines": [
		{
			"node_line_name": "node_line_2",
			"nodes": [
				{
					"node_type": "retrieval",
					"strategy": {
						"metrics": [
							"retrieval_f1",
							"retrieval_recall",
							"retrieval_precision",
						],
					},
					"modules": [{"module_type": "bm25", "top_k": 50}],
				},
				{
					"node_type": "rerank",
					"strategy": {
						"metrics": [
							"retrieval_f1",
							"retrieval_recall",
							"retrieval_precision",
						],
						"speed_threshold": 10,
					},
					"modules": [
						{"module_type": "upr", "model": "llama-2", "havertz": "chelsea"}
					],
				},
			],
		},
		{
			"node_line_name": "node_line_1",
			"nodes": [
				{
					"node_type": "generation",
					"strategy": {
						"metrics": ["bleu", "rouge"],
					},
					"modules": [{"module_type": "gpt-4", "top_p": 0.9}],
				}
			],
		},
	]
}


@pytest.fixture
def pseudo_trial_path():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		os.makedirs(os.path.join(project_dir, "resources"))
		vectordb_config_path = os.path.join(project_dir, "resources", "vectordb.yaml")
		with open(vectordb_config_path, "w") as f:
			yaml.safe_dump(
				{
					"vectordb": [
						{
							"name": "default",
							"db_type": "chroma",
							"client_type": "persistent",
							"embedding_model": "openai",
							"collection_name": "openai",
							"path": os.path.join(project_dir, "resources", "chroma"),
						}
					]
				},
				f,
			)

		trial_path = os.path.join(project_dir, "0")
		os.makedirs(trial_path)
		summary_df.to_csv(os.path.join(trial_path, "summary.csv"), index=False)
		with open(os.path.join(trial_path, "config.yaml"), "w") as f:
			yaml.safe_dump(solution_dict, f)
		yield trial_path


def test_extract_node_line_names(full_config):
	node_line_names = extract_node_line_names(full_config)
	assert node_line_names == [
		"pre_retrieve_node_line",
		"retrieve_node_line",
		"post_retrieve_node_line",
	]


def test_extract_node_strategy(full_config):
	node_strategies = extract_node_strategy(full_config)
	assert set(list(node_strategies.keys())) == {
		"query_expansion",
		"retrieval",
		"passage_reranker",
		"prompt_maker",
		"generator",
	}
	assert node_strategies["retrieval"] == {
		"metrics": ["retrieval_f1", "retrieval_recall", "retrieval_precision"],
		"speed_threshold": 10,
	}


def test_summary_df_to_yaml():
	yaml_dict = summary_df_to_yaml(summary_df, solution_dict)
	assert yaml_dict == solution_dict


def test_extract_best_config(pseudo_trial_path):
	yaml_dict = extract_best_config(pseudo_trial_path)
	assert yaml_dict["node_lines"] == solution_dict["node_lines"]
	with tempfile.NamedTemporaryFile(
		suffix="yaml", mode="w+t", delete=False
	) as yaml_path:
		yaml_dict = extract_best_config(pseudo_trial_path, yaml_path.name)
		assert yaml_dict["node_lines"] == solution_dict["node_lines"]
		assert os.path.exists(yaml_path.name)
		yaml_dict = yaml.safe_load(yaml_path)
		assert yaml_dict["node_lines"] == solution_dict["node_lines"]
		assert yaml_dict["vectordb"][0]["name"] == "default"
		assert yaml_dict["vectordb"][0]["db_type"] == "chroma"
		assert yaml_dict["vectordb"][0]["client_type"] == "persistent"
		assert yaml_dict["vectordb"][0]["embedding_model"] == "openai"
		assert yaml_dict["vectordb"][0]["collection_name"] == "openai"
		yaml_path.close()
		os.unlink(yaml_path.name)


def test_runner(evaluator):
	evaluator.start_trial(os.path.join(resource_dir, "simple_mock.yaml"))
	project_dir = evaluator.project_dir

	def runner_test(runner: Runner):
		answer = runner.run(
			"What is the best movie in Korea? Have Korea movie ever won Oscar?",
			"retrieved_contents",
		)
		assert len(answer) == 10
		assert isinstance(answer, list)
		assert isinstance(answer[0], str)

	runner = Runner.from_trial_folder(os.path.join(project_dir, "0"))
	runner_test(runner)
	runner_test(runner)

	with tempfile.NamedTemporaryFile(
		suffix="yaml", mode="w+t", delete=False
	) as yaml_path:
		extract_best_config(os.path.join(project_dir, "0"), yaml_path.name)
		runner = Runner.from_yaml(yaml_path.name, project_dir=project_dir)
		runner_test(runner)
		yaml_path.close()
		os.unlink(yaml_path.name)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_runner_full(evaluator):
	runner = Runner.from_trial_folder(os.path.join(resource_dir, "result_project", "0"))
	answer = runner.run(
		"What is the best movie in Korea? Have Korea movie ever won Oscar?"
	)
	assert isinstance(answer, str)
	assert bool(answer)


def test_runner_api_server(evaluator):
	project_dir = evaluator.project_dir
	evaluator.start_trial(os.path.join(resource_dir, "simple_mock.yaml"))
	runner = ApiRunner.from_trial_folder(os.path.join(project_dir, "0"))

	client = runner.app.test_client()

	async def post_to_server():
		# Use the TestClient to make a request to the server
		response = await client.post(
			"/v1/run",
			json={
				"query": "What is the best movie in Korea? Have Korea movie ever won Oscar?",
				"result_column": "retrieved_contents",
			},
		)
		json_response = await response.get_json()
		return json_response, response.status_code

	nest_asyncio.apply()

	response_json, response_status_code = asyncio.run(post_to_server())
	assert response_status_code == 200
	assert "result" in response_json
	retrieved_contents = response_json["result"]
	assert len(retrieved_contents) == 10
	assert isinstance(retrieved_contents, list)
	assert isinstance(retrieved_contents[0], str)

	retrieved_contents = response_json["retrieved_passage"]
	assert len(retrieved_contents) == 10
	assert isinstance(retrieved_contents[0]["content"], str)
	assert isinstance(retrieved_contents[0]["doc_id"], str)
	assert retrieved_contents[0]["filepath"] is None
	assert retrieved_contents[0]["file_page"] is None
	assert retrieved_contents[0]["start_idx"] is None
	assert retrieved_contents[0]["end_idx"] is None

	async def post_to_server_retrieve():
		response = await client.post(
			"/v1/retrieve", json={"query": "I have a headache."}
		)
		json_response = await response.get_json()
		return json_response, response.status_code

	response_json, response_status_code = asyncio.run(post_to_server_retrieve())
	assert response_status_code == 200
	assert "passages" in response_json
	passages = response_json["passages"]
	assert len(passages) == 10
	assert "doc_id" in passages[0]
	assert "content" in passages[0]
	assert "score" in passages[0]
	assert isinstance(passages[0]["doc_id"], str)
	assert isinstance(passages[0]["content"], str)
	assert isinstance(passages[0]["score"], float)


@pytest.mark.skip(reason="This test is not working")
def test_runner_api_server_stream(evaluator_trial_done):
	project_dir = evaluator_trial_done.project_dir
	runner = ApiRunner.from_trial_folder(os.path.join(project_dir, "0"))
	client = runner.app.test_client()

	async def post_to_server():
		# Use the TestClient to make a request to the server
		async with client.request(
			"/v1/stream",
			method="POST",
			headers={"Content-Type": "application/json"},
			query_string={
				"query": "What is the best movie in Korea? Have Korea movie ever won Oscar?",
			},
		) as connection:
			response = await connection.receive()
			# Ensure the response status code is 200
			assert connection.status_code == 200

			# Collect streamed data
			streamed_data = []
			async for data in response.body:
				streamed_data.append(data)

	nest_asyncio.apply()
	asyncio.run(post_to_server())
