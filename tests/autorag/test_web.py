import os
import pathlib
import subprocess

import pytest


from autorag.deploy import Runner

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
resource_dir = os.path.join(root_dir, "resources")


@pytest.mark.skip(
	reason="The only way to test it is to run it directly on the web and see if it works."
	"If you want to test it, clear this line and run it."
)
def test_web_cli_yaml_project_dir():
	subprocess.run(
		[
			"autorag",
			"run_web",
			"--yaml_path",
			"../../sample_config/best.yaml",
			"--project_dir",
			"../resources/result_project",
		]
	)


@pytest.mark.skip(
	reason="The only way to test it is to run it directly on the web and see if it works."
	"If you want to test it, clear this line and run it."
)
def test_web_cli_trial():
	subprocess.run(
		["autorag", "run_web", "--trial_path", "../resources/result_project/0"]
	)


@pytest.mark.skip(
	reason="The only way to test it is to run it directly on the web and see if it works."
	"If you want to test it, clear this line and run it."
)
def test_web_library():
	runner = Runner.from_trial_folder(os.path.join(resource_dir, "result_project", "0"))
	runner.run_web()
