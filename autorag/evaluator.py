import glob
import json
import logging
import os
import shutil
from datetime import datetime
from itertools import chain
from typing import List, Dict, Optional
from rich.progress import Progress, BarColumn, TimeElapsedColumn

import pandas as pd
import yaml

from autorag.node_line import run_node_line
from autorag.nodes.retrieval.base import get_bm25_pkl_name
from autorag.nodes.retrieval.bm25 import bm25_ingest
from autorag.nodes.retrieval.vectordb import (
	vectordb_ingest,
	filter_exist_ids,
	filter_exist_ids_from_retrieval_gt,
)
from autorag.schema import Node
from autorag.schema.node import (
	module_type_exists,
	extract_values_from_nodes,
	extract_values_from_nodes_strategy,
)
from autorag.utils import (
	cast_qa_dataset,
	cast_corpus_dataset,
	validate_qa_from_corpus_dataset,
)
from autorag.utils.util import (
	load_summary_file,
	explode,
	load_yaml_config,
	get_event_loop,
)
from autorag.vectordb import load_all_vectordb_from_yaml

logger = logging.getLogger("AutoRAG")

ascii_art = """
                _        _____            _____
     /\        | |      |  __ \     /\   / ____|
    /  \  _   _| |_ ___ | |__) |   /  \ | |  __
   / /\ \| | | | __/ _ \|  _  /   / /\ \| | |_ |
  / ____ \ |_| | || (_) | | \ \  / ____ \ |__| |
 /_/    \_\__,_|\__\___/|_|  \_\/_/    \_\_____|

"""


class Evaluator:
	def __init__(
		self,
		qa_data_path: str,
		corpus_data_path: str,
		project_dir: Optional[str] = None,
	):
		"""
		Initialize an Evaluator object.

		:param qa_data_path: The path to the QA dataset.
		    Must be parquet file.
		:param corpus_data_path: The path to the corpus dataset.
		    Must be parquet file.
		:param project_dir: The path to the project directory.
		    Default is the current directory.
		"""
		# validate data paths
		if not os.path.exists(qa_data_path):
			raise ValueError(f"QA data path {qa_data_path} does not exist.")
		if not os.path.exists(corpus_data_path):
			raise ValueError(f"Corpus data path {corpus_data_path} does not exist.")
		if not qa_data_path.endswith(".parquet"):
			raise ValueError(f"QA data path {qa_data_path} is not a parquet file.")
		if not corpus_data_path.endswith(".parquet"):
			raise ValueError(
				f"Corpus data path {corpus_data_path} is not a parquet file."
			)
		self.qa_data_path = qa_data_path
		self.corpus_data_path = corpus_data_path
		self.qa_data = pd.read_parquet(qa_data_path, engine="pyarrow")
		self.corpus_data = pd.read_parquet(corpus_data_path, engine="pyarrow")
		self.qa_data = cast_qa_dataset(self.qa_data)
		self.corpus_data = cast_corpus_dataset(self.corpus_data)
		self.project_dir = project_dir if project_dir is not None else os.getcwd()
		if not os.path.exists(self.project_dir):
			os.makedirs(self.project_dir)

		validate_qa_from_corpus_dataset(self.qa_data, self.corpus_data)

		# copy dataset to the project directory
		if not os.path.exists(os.path.join(self.project_dir, "data")):
			os.makedirs(os.path.join(self.project_dir, "data"))
		qa_path_in_project = os.path.join(self.project_dir, "data", "qa.parquet")
		if not os.path.exists(qa_path_in_project):
			self.qa_data.to_parquet(qa_path_in_project, index=False)
		corpus_path_in_project = os.path.join(
			self.project_dir, "data", "corpus.parquet"
		)
		if not os.path.exists(corpus_path_in_project):
			self.corpus_data.to_parquet(corpus_path_in_project, index=False)

	def start_trial(
		self, yaml_path: str, skip_validation: bool = False, full_ingest: bool = True
	):
		"""
		Start AutoRAG trial.
		The trial means one experiment to optimize the RAG pipeline.
		It consists of ingesting corpus data, running all nodes and modules, evaluating and finding the optimal modules.

		:param yaml_path: The config YAML path
		:param skip_validation: If True, it skips the validation step.
			The validation step checks the input config YAML file is well formatted,
			and there is any problem with the system settings.
			Default is False.
		:param full_ingest: If True, it checks the whole corpus data from corpus.parquet that exists in the Vector DB.
			If your corpus is huge and don't want to check the whole vector DB, please set it to False.
		:return: None
		"""
		# Make Resources directory
		os.makedirs(os.path.join(self.project_dir, "resources"), exist_ok=True)

		if not skip_validation:
			logger.info(ascii_art)
			logger.info(
				"Start Validation input data and config YAML file first. "
				"If you want to skip this, put the --skip_validation flag or "
				"`skip_validation` at the start_trial function."
			)
			from autorag.validator import Validator  # resolve circular import

			validator = Validator(
				qa_data_path=self.qa_data_path, corpus_data_path=self.corpus_data_path
			)
			validator.validate(yaml_path)

		os.environ["PROJECT_DIR"] = self.project_dir

		trial_name = self.__get_new_trial_name()
		self.__make_trial_dir(trial_name)

		# copy YAML file to the trial directory
		shutil.copy(
			yaml_path, os.path.join(self.project_dir, trial_name, "config.yaml")
		)
		yaml_dict = load_yaml_config(yaml_path)
		vectordb = yaml_dict.get("vectordb", [])

		vectordb_config_path = os.path.join(
			self.project_dir, "resources", "vectordb.yaml"
		)
		with open(vectordb_config_path, "w") as f:
			yaml.safe_dump({"vectordb": vectordb}, f)

		node_lines = self._load_node_lines(yaml_path)
		self.__ingest_bm25_full(node_lines)

		with Progress(
			"[progress.description]{task.description}",
			BarColumn(),
			"[progress.percentage]{task.percentage:>3.0f}%",
			"[progress.bar]{task.completed}/{task.total}",
			TimeElapsedColumn(),
		) as progress:
			# Ingest VectorDB corpus
			if any(
				list(
					map(
						lambda nodes: module_type_exists(nodes, "vectordb"),
						node_lines.values(),
					)
				)
			):
				task_ingest = progress.add_task("[cyan]Ingesting VectorDB...", total=1)

				loop = get_event_loop()
				loop.run_until_complete(self.__ingest_vectordb(yaml_path, full_ingest))

				progress.update(task_ingest, completed=1)

			trial_summary_df = pd.DataFrame(
				columns=[
					"node_line_name",
					"node_type",
					"best_module_filename",
					"best_module_name",
					"best_module_params",
					"best_execution_time",
				]
			)
			task_eval = progress.add_task(
				"[cyan]Evaluating...", total=sum(map(len, node_lines.values()))
			)

			for i, (node_line_name, node_line) in enumerate(node_lines.items()):
				node_line_dir = os.path.join(
					self.project_dir, trial_name, node_line_name
				)
				os.makedirs(node_line_dir, exist_ok=False)
				if i == 0:
					previous_result = self.qa_data
				logger.info(f"Running node line {node_line_name}...")
				previous_result = run_node_line(
					node_line, node_line_dir, previous_result, progress, task_eval
				)

				trial_summary_df = self._append_node_line_summary(
					node_line_name, node_line_dir, trial_summary_df
				)

			trial_summary_df.to_csv(
				os.path.join(self.project_dir, trial_name, "summary.csv"), index=False
			)

			logger.info("Evaluation complete.")

	def __ingest_bm25_full(self, node_lines: Dict[str, List[Node]]):
		if any(
			list(
				map(
					lambda nodes: module_type_exists(nodes, "bm25"), node_lines.values()
				)
			)
		):
			logger.info("Embedding BM25 corpus...")
			bm25_tokenizer_list = list(
				chain.from_iterable(
					map(
						lambda nodes: self._find_bm25_tokenizer(nodes),
						node_lines.values(),
					)
				)
			)

			if len(bm25_tokenizer_list) == 0:
				bm25_tokenizer_list = ["porter_stemmer"]
			for bm25_tokenizer in bm25_tokenizer_list:
				bm25_dir = os.path.join(
					self.project_dir, "resources", get_bm25_pkl_name(bm25_tokenizer)
				)
				if not os.path.exists(os.path.dirname(bm25_dir)):
					os.makedirs(os.path.dirname(bm25_dir))
				# ingest because bm25 supports update new corpus data
				bm25_ingest(bm25_dir, self.corpus_data, bm25_tokenizer=bm25_tokenizer)
			logger.info("BM25 corpus embedding complete.")

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

	@staticmethod
	def _load_node_lines(yaml_path: str) -> Dict[str, List[Node]]:
		yaml_dict = load_yaml_config(yaml_path)
		node_lines = yaml_dict["node_lines"]
		node_line_dict = {}
		for node_line in node_lines:
			node_line_dict[node_line["node_line_name"]] = list(
				map(lambda node: Node.from_dict(node), node_line["nodes"])
			)
		return node_line_dict

	def restart_trial(self, trial_path: str):
		logger.info(ascii_art)
		os.environ["PROJECT_DIR"] = self.project_dir
		# Check if trial_path exists
		if not os.path.exists(trial_path):
			raise ValueError(f"Trial path {trial_path} does not exist.")
		# Check if trial is completed
		if os.path.exists(os.path.join(trial_path, "summary.csv")):
			raise ValueError(f"Trial path {trial_path} is already completed.")

		# Extract node lines from config.yaml
		yaml_path = os.path.join(trial_path, "config.yaml")
		node_lines = self._load_node_lines(yaml_path)

		node_line_names = list(node_lines.keys())
		nodes = list(node_lines.values())
		node_names = list(
			map(lambda node: list(map(lambda n: n.node_type, node)), nodes)
		)

		# If the First Node Line folder hasn't even been created, proceed to start_trial
		if not os.path.exists(os.path.join(trial_path, node_line_names[0])):
			self.start_trial(yaml_path)
			return None

		# Find conflict node line and node
		conflict_line_name, conflict_node_name = self.__find_conflict_point(
			trial_path, node_line_names, node_lines
		)
		node_dir = os.path.join(trial_path, conflict_line_name, conflict_node_name)
		if os.path.exists(node_dir):
			shutil.rmtree(node_dir)

		# Set remain_nodes and remain_lines
		remain_nodes, completed_node_names, remain_lines, remain_line_names = (
			self._set_remain_nodes_and_lines(
				node_line_names,
				nodes,
				node_names,
				conflict_node_name,
				conflict_line_name,
			)
		)
		# Set previous_result
		previous_result = self.__set_previous_result(
			node_line_names, node_names, trial_path, conflict_node_name
		)

		# Run Node
		if remain_nodes:
			conflict_line_dir = os.path.join(trial_path, conflict_line_name)
			summary_lst = []
			# Get already run node summary and append to summary_lst
			for completed_node_name in completed_node_names:
				summary_lst = self._append_node_summary(
					conflict_line_dir, completed_node_name, summary_lst
				)
			for node in remain_nodes:
				previous_result = node.run(previous_result, conflict_line_dir)
				summary_lst = self._append_node_summary(
					conflict_line_dir, node.node_type, summary_lst
				)
			pd.DataFrame(summary_lst).to_csv(
				os.path.join(conflict_line_dir, "summary.csv"), index=False
			)

		# Run node line
		trial_summary_df = pd.DataFrame(
			columns=[
				"node_line_name",
				"node_type",
				"best_module_filename",
				"best_module_name",
				"best_module_params",
				"best_execution_time",
			]
		)
		completed_line_names = node_line_names[
			: node_line_names.index(conflict_line_name)
		]
		# Get already run node line's summary and append to trial_summary_df
		if completed_line_names:
			for line_name in completed_line_names:
				node_line_dir = os.path.join(trial_path, line_name)
				trial_summary_df = self._append_node_line_summary(
					line_name, node_line_dir, trial_summary_df
				)
		if remain_lines:
			for node_line_name, node_line in zip(remain_line_names, remain_lines):
				node_line_dir = os.path.join(trial_path, node_line_name)
				if not os.path.exists(node_line_dir):
					os.makedirs(node_line_dir)
				logger.info(f"Running node line {node_line_name}...")
				previous_result = run_node_line(
					node_line, node_line_dir, previous_result
				)
				trial_summary_df = self._append_node_line_summary(
					node_line_name, node_line_dir, trial_summary_df
				)
		trial_summary_df.to_csv(os.path.join(trial_path, "summary.csv"), index=False)

		logger.info("Evaluation complete.")

	def __find_conflict_point(
		self,
		trial_path: str,
		node_line_names: List[str],
		node_lines: Dict[str, List[Node]],
	) -> tuple[str, str]:
		for node_line_name in node_line_names:
			node_line_dir = os.path.join(trial_path, node_line_name)
			if not os.path.exists(node_line_dir):
				return node_line_name, node_lines[node_line_name][0].node_type

			if not os.path.exists(os.path.join(node_line_dir, "summary.csv")):
				conflict_node_name = self._find_conflict_node_name(
					node_line_dir, node_lines[node_line_name]
				)
				return node_line_name, conflict_node_name

		raise ValueError(f"No error node line found in {trial_path}.")

	@staticmethod
	def _find_conflict_node_name(node_line_dir: str, node_line: List[Node]) -> str:
		for node in node_line:
			node_dir = os.path.join(node_line_dir, node.node_type)
			if not os.path.exists(node_dir) or not os.path.exists(
				os.path.join(node_dir, "summary.csv")
			):
				return node.node_type
		raise TypeError("No conflict node name found.")

	def __set_previous_result(
		self,
		node_line_names: List[str],
		node_names: List[List[str]],
		trial_path: str,
		conflict_node_name: str,
	):
		exploded_node_line, exploded_node = explode(node_line_names, node_names)
		conflict_node_index = exploded_node.index(conflict_node_name)
		# Set previous_result
		if conflict_node_index == 0:
			previous_result = self.qa_data
		else:
			previous_node_line = exploded_node_line[conflict_node_index - 1]
			previous_node = exploded_node[conflict_node_index - 1]

			previous_node_dir = os.path.join(
				trial_path, previous_node_line, previous_node
			)
			best_file_pattern = f"{previous_node_dir}/best_*.parquet"
			previous_result = pd.read_parquet(
				glob.glob(best_file_pattern)[0], engine="pyarrow"
			)
		return previous_result

	@staticmethod
	def _set_remain_nodes_and_lines(
		node_line_names: List[str],
		nodes: List[List[Node]],
		node_names: List[List[str]],
		conflict_node_name: str,
		conflict_node_line_name: str,
	):
		conflict_node_line_index = node_line_names.index(conflict_node_line_name)
		full_conflict_node_line_nodes = nodes[conflict_node_line_index]
		full_conflict_node_line_node_names = node_names[conflict_node_line_index]

		if conflict_node_name == full_conflict_node_line_node_names[0]:
			remain_nodes = None
			completed_node_names = None
			remain_node_lines = nodes[conflict_node_line_index:]
			remain_node_line_names = node_line_names[conflict_node_line_index:]
		else:
			conflict_node_index = full_conflict_node_line_node_names.index(
				conflict_node_name
			)
			remain_nodes = full_conflict_node_line_nodes[conflict_node_index:]
			completed_node_names = full_conflict_node_line_node_names[
				:conflict_node_index
			]
			if conflict_node_line_index + 1 >= len(node_line_names):
				remain_node_lines = None
				remain_node_line_names = None
			else:
				remain_node_lines = nodes[conflict_node_line_index + 1 :]
				remain_node_line_names = node_line_names[conflict_node_line_index + 1 :]
		return (
			remain_nodes,
			completed_node_names,
			remain_node_lines,
			remain_node_line_names,
		)

	@staticmethod
	def _append_node_line_summary(
		node_line_name: str, node_line_dir: str, trial_summary_df: pd.DataFrame
	):
		summary_df = load_summary_file(
			os.path.join(node_line_dir, "summary.csv"),
			dict_columns=["best_module_params"],
		)
		summary_df = summary_df.assign(node_line_name=node_line_name)
		summary_df = summary_df[list(trial_summary_df.columns)]
		if len(trial_summary_df) <= 0:
			trial_summary_df = summary_df
		else:
			trial_summary_df = pd.concat(
				[trial_summary_df, summary_df], ignore_index=True
			)
		return trial_summary_df

	@staticmethod
	def _append_node_summary(
		node_line_dir: str, node_name: str, summary_lst: List[Dict]
	):
		node_summary_df = load_summary_file(
			os.path.join(node_line_dir, node_name, "summary.csv")
		)
		best_node_row = node_summary_df.loc[node_summary_df["is_best"]]
		summary_lst.append(
			{
				"node_type": node_name,
				"best_module_filename": best_node_row["filename"].values[0],
				"best_module_name": best_node_row["module_name"].values[0],
				"best_module_params": best_node_row["module_params"].values[0],
				"best_execution_time": best_node_row["execution_time"].values[0],
			}
		)
		return summary_lst

	@staticmethod
	def _find_bm25_tokenizer(nodes: List[Node]):
		bm25_tokenizer_list = extract_values_from_nodes(nodes, "bm25_tokenizer")
		strategy_tokenizer_list = list(
			chain.from_iterable(
				extract_values_from_nodes_strategy(nodes, "bm25_tokenizer")
			)
		)
		return list(set(bm25_tokenizer_list + strategy_tokenizer_list))

	@staticmethod
	def _find_embedding_model(nodes: List[Node]):
		embedding_models_list = extract_values_from_nodes(nodes, "embedding_model")
		retrieval_module_dicts = extract_values_from_nodes_strategy(
			nodes, "retrieval_modules"
		)
		for retrieval_modules in retrieval_module_dicts:
			vectordb_modules = list(
				filter(lambda x: x["module_type"] == "vectordb", retrieval_modules)
			)
			embedding_models_list.extend(
				list(map(lambda x: x.get("embedding_model", None), vectordb_modules))
			)
		embedding_models_list = list(
			filter(lambda x: x is not None, embedding_models_list)
		)
		return list(set(embedding_models_list))

	async def __ingest_vectordb(self, yaml_path, full_ingest: bool):
		vectordb_list = load_all_vectordb_from_yaml(yaml_path, self.project_dir)
		if full_ingest is True:
			# get the target ingest corpus from the whole corpus
			for vectordb in vectordb_list:
				target_corpus = await filter_exist_ids(vectordb, self.corpus_data)
				await vectordb_ingest(vectordb, target_corpus)
		else:
			# get the target ingest corpus from the retrieval gt only
			for vectordb in vectordb_list:
				target_corpus = await filter_exist_ids_from_retrieval_gt(
					vectordb, self.qa_data, self.corpus_data
				)
				await vectordb_ingest(vectordb, target_corpus)
