import logging
import os
import pathlib
import uuid
from copy import deepcopy
from typing import Optional, Dict, List

import pandas as pd
import yaml

from autorag.support import get_support_modules
from autorag.utils.util import load_summary_file, load_yaml_config

logger = logging.getLogger("AutoRAG")


def extract_node_line_names(config_dict: Dict) -> List[str]:
	"""
	Extract node line names with the given config dictionary order.

	:param config_dict: The YAML configuration dict for the pipeline.
	    You can load this to access trail_folder/config.yaml.
	:return: The list of node line names.
	    It is the order of the node line names in the pipeline.
	"""
	return [node_line["node_line_name"] for node_line in config_dict["node_lines"]]


def extract_node_strategy(config_dict: Dict) -> Dict:
	"""
	Extract node strategies with the given config dictionary.
	The return value is a dictionary of the node type and its strategy.

	:param config_dict: The YAML configuration dict for the pipeline.
	    You can load this to access trail_folder/config.yaml.
	:return: Key is node_type and value is strategy dict.
	"""
	return {
		node["node_type"]: node.get("strategy", {})
		for node_line in config_dict["node_lines"]
		for node in node_line["nodes"]
	}


def summary_df_to_yaml(summary_df: pd.DataFrame, config_dict: Dict) -> Dict:
	"""
	Convert trial summary dataframe to config yaml file.

	:param summary_df: The trial summary dataframe of the evaluated trial.
	:param config_dict: The yaml configuration dict for the pipeline.
	    You can load this to access trail_folder/config.yaml.
	:return: Dictionary of config yaml file.
	    You can save this dictionary to yaml file.
	"""

	# summary_df columns : 'node_line_name', 'node_type', 'best_module_filename',
	#                      'best_module_name', 'best_module_params', 'best_execution_time'
	node_line_names = extract_node_line_names(config_dict)
	node_strategies = extract_node_strategy(config_dict)
	strategy_df = pd.DataFrame(
		{
			"node_type": list(node_strategies.keys()),
			"strategy": list(node_strategies.values()),
		}
	)
	summary_df = summary_df.merge(strategy_df, on="node_type", how="left")
	summary_df["categorical_node_line_name"] = pd.Categorical(
		summary_df["node_line_name"], categories=node_line_names, ordered=True
	)
	summary_df = summary_df.sort_values(by="categorical_node_line_name")
	grouped = summary_df.groupby("categorical_node_line_name", observed=False)

	node_lines = [
		{
			"node_line_name": node_line_name,
			"nodes": [
				{
					"node_type": row["node_type"],
					"strategy": row["strategy"],
					"modules": [
						{
							"module_type": row["best_module_name"],
							**row["best_module_params"],
						}
					],
				}
				for _, row in node_line.iterrows()
			],
		}
		for node_line_name, node_line in grouped
	]
	return {"node_lines": node_lines}


def extract_best_config(trial_path: str, output_path: Optional[str] = None) -> Dict:
	"""
	Extract the optimal pipeline from the evaluated trial.

	:param trial_path: The path to the trial directory that you want to extract the pipeline from.
	    Must already be evaluated.
	:param output_path: Output path that pipeline yaml file will be saved.
	    Must be .yaml or .yml file.
	    If None, it does not save YAML file and just returns dict values.
	    Default is None.
	:return: The dictionary of the extracted pipeline.
	"""
	summary_path = os.path.join(trial_path, "summary.csv")
	if not os.path.exists(summary_path):
		raise ValueError(f"summary.csv does not exist in {trial_path}.")
	trial_summary_df = load_summary_file(
		summary_path, dict_columns=["best_module_params"]
	)
	config_yaml_path = os.path.join(trial_path, "config.yaml")
	with open(config_yaml_path, "r") as f:
		config_dict = yaml.safe_load(f)
	yaml_dict = summary_df_to_yaml(trial_summary_df, config_dict)
	yaml_dict["vectordb"] = extract_vectordb_config(trial_path)
	if output_path is not None:
		with open(output_path, "w") as f:
			yaml.safe_dump(yaml_dict, f)
	return yaml_dict


def extract_vectordb_config(trial_path: str) -> List[Dict]:
	# get vectordb.yaml file
	project_dir = pathlib.PurePath(os.path.realpath(trial_path)).parent
	vectordb_config_path = os.path.join(project_dir, "resources", "vectordb.yaml")
	if not os.path.exists(vectordb_config_path):
		raise ValueError(f"vectordb.yaml does not exist in {vectordb_config_path}.")
	with open(vectordb_config_path, "r") as f:
		vectordb_dict = yaml.safe_load(f)
	result = vectordb_dict.get("vectordb", [])
	if len(result) != 0:
		return result
	# return default setting of chroma
	return [
		{
			"name": "default",
			"db_type": "chroma",
			"client_type": "persistent",
			"embedding_model": "openai",
			"collection_name": "openai",
			"path": os.path.join(project_dir, "resources", "chroma"),
		}
	]


class BaseRunner:
	def __init__(self, config: Dict, project_dir: Optional[str] = None):
		self.config = config
		project_dir = os.getcwd() if project_dir is None else project_dir
		os.environ["PROJECT_DIR"] = project_dir

		# init modules
		node_lines = deepcopy(self.config["node_lines"])
		self.module_instances = []
		self.module_params = []
		for node_line in node_lines:
			for node in node_line["nodes"]:
				if len(node["modules"]) != 1:
					raise ValueError(
						"The number of modules in a node must be 1 for using runner."
						"Please use extract_best_config method for extracting yaml file from evaluated trial."
					)
				module = node["modules"][0]
				module_type = module.pop("module_type")
				module_params = module
				module_instance = get_support_modules(module_type)(
					project_dir=project_dir,
					**module_params,
				)
				self.module_instances.append(module_instance)
				self.module_params.append(module_params)

	@classmethod
	def from_yaml(cls, yaml_path: str, project_dir: Optional[str] = None):
		"""
		Load Runner from the YAML file.
		Must be extracted YAML file from the evaluated trial using the extract_best_config method.

		:param yaml_path: The path of the YAML file.
		:param project_dir: The path of the project directory.
			Default is the current directory.
		:return: Initialized Runner.
		"""
		config = load_yaml_config(yaml_path)
		return cls(config, project_dir=project_dir)

	@classmethod
	def from_trial_folder(cls, trial_path: str):
		"""
		Load Runner from the evaluated trial folder.
		Must already be evaluated using Evaluator class.
		It sets the project_dir as the parent directory of the trial folder.

		:param trial_path: The path of the trial folder.
		:return: Initialized Runner.
		"""
		config = extract_best_config(trial_path)
		return cls(config, project_dir=os.path.dirname(trial_path))


class Runner(BaseRunner):
	def run(self, query: str, result_column: str = "generated_texts"):
		"""
		Run the pipeline with query.
		The loaded pipeline must start with a single query,
		so the first module of the pipeline must be `query_expansion` or `retrieval` module.

		:param query: The query of the user.
		:param result_column: The result column name for the answer.
		    Default is `generated_texts`, which is the output of the `generation` module.
		:return: The result of the pipeline.
		"""
		previous_result = pd.DataFrame(
			{
				"qid": str(uuid.uuid4()),
				"query": [query],
				"retrieval_gt": [[]],
				"generation_gt": [""],
			}
		)  # pseudo qa data for execution
		for module_instance, module_param in zip(
			self.module_instances, self.module_params
		):
			new_result = module_instance.pure(
				previous_result=previous_result, **module_param
			)
			duplicated_columns = previous_result.columns.intersection(
				new_result.columns
			)
			drop_previous_result = previous_result.drop(columns=duplicated_columns)
			previous_result = pd.concat([drop_previous_result, new_result], axis=1)

		return previous_result[result_column].tolist()[0]
