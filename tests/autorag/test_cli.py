import os
import pathlib
import subprocess
import tempfile
from distutils.dir_util import copy_tree

from click.testing import CliRunner

from autorag.cli import cli

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
resource_dir = os.path.join(root_dir, "resources")


def test_evaluator_cli():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		result = subprocess.run(
			[
				"autorag",
				"evaluate",
				"--config",
				os.path.join(resource_dir, "simple_mock.yaml"),
				"--qa_data_path",
				os.path.join(resource_dir, "qa_data_sample.parquet"),
				"--corpus_data_path",
				os.path.join(resource_dir, "corpus_data_sample.parquet"),
				"--project_dir",
				project_dir,
				"--skip_validation",
				"true",
			]
		)
		assert result.returncode == 0
		# check if the files are created
		assert os.path.exists(os.path.join(project_dir, "0"))
		assert os.path.exists(os.path.join(project_dir, "data"))
		assert os.path.exists(os.path.join(project_dir, "resources"))
		assert os.path.exists(os.path.join(project_dir, "trial.json"))
		assert os.path.exists(os.path.join(project_dir, "0", "retrieve_node_line"))
		assert os.path.exists(
			os.path.join(project_dir, "0", "retrieve_node_line", "retrieval")
		)
		assert os.path.exists(
			os.path.join(
				project_dir, "0", "retrieve_node_line", "retrieval", "0.parquet"
			)
		)


def test_validator_cli():
	result = subprocess.run(
		[
			"autorag",
			"validate",
			"--config",
			os.path.join(resource_dir, "simple_mock.yaml"),
			"--qa_data_path",
			os.path.join(resource_dir, "qa_data_sample.parquet"),
			"--corpus_data_path",
			os.path.join(resource_dir, "corpus_data_sample.parquet"),
		]
	)
	assert result.returncode == 0


def test_run_api():
	# This test code only tests cli function, not API server itself
	# If you are looking for API server test code, please go to test_deploy.py test_runner_api_server()
	runner = CliRunner()
	result = runner.invoke(
		cli,
		[
			"run_api",
			"--config_path",
			"test/path/test.yaml",
			"--host",
			"0.0.0.0",
			"--port",
			"8080",
		],
	)
	assert (
		result.exit_code == 1
	)  # it will occur error because I run this test with a wrong yaml path.
	# But it means that the command is working well. If not, it will occur exit_code 2.


def test_extract_best_config_cli():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		trial_path = os.path.join(resource_dir, "result_project", "0")
		output_path = os.path.join(project_dir, "best.yaml")
		subprocess.run(
			[
				"autorag",
				"extract_best_config",
				"--trial_path",
				trial_path,
				"--output_path",
				output_path,
			]
		)
		assert os.path.exists(output_path)


def test_restart_evaluate():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		original_path = os.path.join(resource_dir, "result_project")
		copy_tree(original_path, project_dir)
		trial_path = os.path.join(project_dir, "1")
		subprocess.run(["autorag", "restart_evaluate", "--trial_path", trial_path])
		assert os.path.exists(os.path.join(trial_path, "summary.csv"))


def test_restart_evaluate_leads_start_evaluate():
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		original_path = os.path.join(resource_dir, "result_project")
		copy_tree(original_path, project_dir)
		trial_path = os.path.join(project_dir, "3")
		subprocess.run(["autorag", "restart_evaluate", "--trial_path", trial_path])
		restart_path = os.path.join(project_dir, "4")
		assert os.path.exists(os.path.join(restart_path, "summary.csv"))
