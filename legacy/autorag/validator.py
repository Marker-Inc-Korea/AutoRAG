import itertools
import logging
import os
import tempfile

import pandas as pd

from autorag.evaluator import Evaluator
from autorag.utils import (
	cast_qa_dataset,
	cast_corpus_dataset,
	validate_qa_from_corpus_dataset,
)

logger = logging.getLogger("AutoRAG")


class Validator:
	def __init__(self, qa_data_path: str, corpus_data_path: str):
		"""
		Initialize a Validator object.

		:param qa_data_path: The path to the QA dataset.
		    Must be parquet file.
		:param corpus_data_path: The path to the corpus dataset.
		    Must be parquet file.
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
		self.qa_data = pd.read_parquet(qa_data_path, engine="pyarrow")
		self.corpus_data = pd.read_parquet(corpus_data_path, engine="pyarrow")
		self.qa_data = cast_qa_dataset(self.qa_data)
		self.corpus_data = cast_corpus_dataset(self.corpus_data)

	def validate(self, yaml_path: str, qa_cnt: int = 5, random_state: int = 42):
		# Determine the sample size and log a warning if qa_cnt is larger than available records
		available_records = len(self.qa_data)
		safe_sample_size = min(qa_cnt, available_records)  # 먼저 safe_sample_size 계산

		if safe_sample_size < qa_cnt:
			logger.warning(
				f"Minimal Requested sample size ({qa_cnt}) is larger than available records ({available_records}). "
				f"Sampling will be limited to {safe_sample_size} records. "
			)

		# safe sample QA data
		sample_qa_df = self.qa_data.sample(
			n=safe_sample_size, random_state=random_state
		)
		sample_qa_df.reset_index(drop=True, inplace=True)

		# get doc_id
		temp_qa_df = sample_qa_df.copy(deep=True)
		flatten_retrieval_gts = (
			temp_qa_df["retrieval_gt"]
			.apply(lambda x: list(itertools.chain.from_iterable(x)))
			.tolist()
		)
		target_doc_ids = list(itertools.chain.from_iterable(flatten_retrieval_gts))

		# make sample corpus data
		sample_corpus_df = self.corpus_data.loc[
			self.corpus_data["doc_id"].isin(target_doc_ids)
		]
		sample_corpus_df.reset_index(drop=True, inplace=True)

		validate_qa_from_corpus_dataset(sample_qa_df, sample_corpus_df)

		# start Evaluate at temp project directory
		with (
			tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as qa_path,
			tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as corpus_path,
			tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_project_dir,
		):
			sample_qa_df.to_parquet(qa_path.name, index=False)
			sample_corpus_df.to_parquet(corpus_path.name, index=False)

			evaluator = Evaluator(
				qa_data_path=qa_path.name,
				corpus_data_path=corpus_path.name,
				project_dir=temp_project_dir,
			)
			evaluator.start_trial(yaml_path, skip_validation=True)
			qa_path.close()
			corpus_path.close()
			os.unlink(qa_path.name)
			os.unlink(corpus_path.name)

		logger.info("Validation complete.")
