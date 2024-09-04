import json
import logging
import os
import shutil
from datetime import datetime
from typing import Optional

from autorag.data.parse.run import run_parser
from autorag.data.utils.util import load_yaml, get_param_combinations

logger = logging.getLogger("AutoRAG")


class Parser:
	def __init__(self, data_path_glob: str, project_dir: Optional[str] = None):
		self.data_path_glob = data_path_glob
		self.project_dir = project_dir if project_dir is not None else os.getcwd()

	def start_parsing(self, yaml_path: str):
		trial_name = self.__get_new_trial_name()
		self.__make_trial_dir(trial_name)

		# copy yaml file to trial directory
		shutil.copy(
			yaml_path, os.path.join(self.project_dir, trial_name, "parse_config.yaml")
		)

		# load yaml file
		modules = load_yaml(yaml_path)

		input_modules, input_params = get_param_combinations(modules)

		logger.info("Parsing Start...")
		run_parser(
			modules=input_modules,
			module_params=input_params,
			data_path_glob=self.data_path_glob,
			trial_path=os.path.join(self.project_dir, trial_name),
		)
		logger.info("Parsing Done!")

	def __get_new_trial_name(self) -> str:
		trial_json_path = os.path.join(self.project_dir, "trial.json")
		if not os.path.exists(trial_json_path):
			return "0"
		with open(trial_json_path, "r") as f:
			trial_json = json.load(f)
		return str(int(trial_json[-1]["trial_name"]) + 1)

	def __make_trial_dir(self, trial_name: str):
		trial_json_path = os.path.join(self.project_dir, "trial.json")
		if os.path.exists(trial_json_path):
			with open(trial_json_path, "r") as f:
				trial_json = json.load(f)
		else:
			trial_json = []

		trial_json.append(
			{
				"trial_name": trial_name,
				"start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
			}
		)
		os.makedirs(os.path.join(self.project_dir, trial_name))
		with open(trial_json_path, "w") as f:
			json.dump(trial_json, f, indent=4)
