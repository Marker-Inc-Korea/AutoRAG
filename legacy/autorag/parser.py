import logging
import os
import shutil
from typing import Optional

from autorag.data.parse.run import run_parser
from autorag.data.utils.util import load_yaml, get_param_combinations

logger = logging.getLogger("AutoRAG")


class Parser:
	def __init__(self, data_path_glob: str, project_dir: Optional[str] = None):
		self.data_path_glob = data_path_glob
		self.project_dir = project_dir if project_dir is not None else os.getcwd()

	def start_parsing(self, yaml_path: str, all_files: bool = False):
		if not os.path.exists(self.project_dir):
			os.makedirs(self.project_dir)

		# copy yaml file to project directory
		shutil.copy(yaml_path, os.path.join(self.project_dir, "parse_config.yaml"))

		# load yaml file
		modules = load_yaml(yaml_path)

		input_modules, input_params = get_param_combinations(modules)

		logger.info("Parsing Start...")
		run_parser(
			modules=input_modules,
			module_params=input_params,
			data_path_glob=self.data_path_glob,
			project_dir=self.project_dir,
			all_files=all_files,
		)
		logger.info("Parsing Done!")
