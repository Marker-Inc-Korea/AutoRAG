import logging
import os
import shutil
from typing import Optional

import pandas as pd

from autorag.data.chunk.run import run_chunker
from autorag.data.utils.util import load_yaml, get_param_combinations

logger = logging.getLogger("AutoRAG")


class Chunker:
	def __init__(self, raw_df: pd.DataFrame, project_dir: Optional[str] = None):
		self.parsed_raw = raw_df
		self.project_dir = project_dir if project_dir is not None else os.getcwd()

	@classmethod
	def from_parquet(
		cls, parsed_data_path: str, project_dir: Optional[str] = None
	) -> "Chunker":
		if not os.path.exists(parsed_data_path):
			raise ValueError(f"parsed_data_path {parsed_data_path} does not exist.")
		if not parsed_data_path.endswith("parquet"):
			raise ValueError(
				f"parsed_data_path {parsed_data_path} is not a parquet file."
			)
		parsed_result = pd.read_parquet(parsed_data_path, engine="pyarrow")
		return cls(parsed_result, project_dir)

	def start_chunking(self, yaml_path: str):
		if not os.path.exists(self.project_dir):
			os.makedirs(self.project_dir)

		# Copy YAML file to the trial directory
		shutil.copy(yaml_path, os.path.join(self.project_dir, "chunk_config.yaml"))

		# load yaml file
		modules = load_yaml(yaml_path)

		input_modules, input_params = get_param_combinations(modules)

		logger.info("Chunking Start...")
		run_chunker(
			modules=input_modules,
			module_params=input_params,
			parsed_result=self.parsed_raw,
			project_dir=self.project_dir,
		)
		logger.info("Chunking Done!")
