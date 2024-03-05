import os
import pathlib
import subprocess

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent
resource_dir = os.path.join(root_dir, 'resources')


def test_web_cli_yaml():
    result = subprocess.run(
        ['autorag', 'run_web', '--yaml_path', 'test/path/test.yaml', '--project_dir', 'test/path'])