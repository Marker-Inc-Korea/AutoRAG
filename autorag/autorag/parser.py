import logging
import os
import shutil
from typing import Optional, Dict

from autorag.data.parse.run import run_parser
from autorag.data.utils.util import load_yaml, get_param_combinations

import pandas as pd
import tempfile
import glob
from pathlib import Path

logger = logging.getLogger("AutoRAG")


class Parser:
	def __init__(self, data_path_glob: str, project_dir: Optional[str] = None):
		self.data_path_glob = data_path_glob
		self.project_dir = Path(project_dir or Path.cwd()).expanduser().resolve()
		self._temp_dir: Optional[Path] = None
		self._path_map: Dict[str, str] = {}

	def start_parsing(
		self, yaml_path: str, all_files: bool = False, recursive: bool = False
	):
		if not os.path.exists(self.project_dir):
			os.makedirs(self.project_dir)

		# copy yaml file to project directory
		shutil.copy(yaml_path, os.path.join(self.project_dir, "parse_config.yaml"))

		# load yaml file
		modules = load_yaml(yaml_path)

		input_modules, input_params = get_param_combinations(modules)

		data_glob = self._prepare_temp_flat_dir() if recursive else self.data_path_glob

		try:
			logger.info("Parsing Start...")
			run_parser(
				modules=input_modules,
				module_params=input_params,
				data_path_glob=data_glob,
				project_dir=self.project_dir,
				all_files=all_files,
			)
			logger.info("Parsing Done!")
			self._rewrite_output_paths()
		finally:
			self._cleanup_temp_dir()

	# 1) Build flat temp dir
	def _prepare_temp_flat_dir(self) -> str:
		matching_files = glob.glob(self.data_path_glob, recursive=True)
		if not matching_files:
			raise FileNotFoundError(
				f"No files matched recursively for pattern: {self.data_path_glob}"
			)

		self._temp_dir = Path(tempfile.mkdtemp(prefix="autorag_flat_"))

		for i, src in enumerate(matching_files):
			src_path = Path(src)
			rel = src_path.relative_to(Path(os.path.commonpath(matching_files)))
			flat_name = "__".join(rel.parts)
			dst = self._temp_dir / f"{i:06d}__{flat_name}"
			shutil.copy2(src_path, dst)
			self._path_map[str(dst)] = str(src_path)

		return str(self._temp_dir / "*.md")

	# 2) After parsing, rewrite paths in AutoRAG
	def _rewrite_output_paths(self) -> None:
		"""Replace temp paths with original ones in output Parquet/JSON files."""
		# Parquet
		for parquet in self.project_dir.rglob("*.parquet"):
			self._patch_dataframe(parquet, fmt="parquet")
		# JSONL
		for jsonl in self.project_dir.rglob("*.jsonl"):
			self._patch_dataframe(jsonl, fmt="jsonl")

	def _patch_dataframe(self, file: Path, *, fmt: str) -> None:
		cols_to_patch = {"source", "file_path", "path"}
		if fmt == "parquet":
			df = pd.read_parquet(file)
		else:  # jsonl
			df = pd.read_json(file, lines=True)

		intersect = cols_to_patch & set(df.columns)
		if not intersect:
			return

		for col in intersect:
			df[col] = df[col].map(lambda p: self._path_map.get(p, p))

		if fmt == "parquet":
			df.to_parquet(file, index=False)
		else:
			df.to_json(file, orient="records", lines=True)

		logger.debug("Re‑wrote paths in %s", file)

	# 3) Cleanup
	def _cleanup_temp_dir(self) -> None:
		if self._temp_dir and self._temp_dir.exists():
			shutil.rmtree(self._temp_dir)
			logger.debug("Temp directory removed: %s", self._temp_dir)
