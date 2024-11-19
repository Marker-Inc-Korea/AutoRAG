import abc
import logging
from pathlib import Path
from typing import Union

import pandas as pd

from autorag.schema.base import BaseModule
from autorag.utils import validate_qa_dataset

logger = logging.getLogger("AutoRAG")


class BasePassageFilter(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs):
		logger.info(f"Initialize passage filter node - {self.__class__.__name__}")

	def __del__(self):
		logger.info(f"Prompt maker node - {self.__class__.__name__} module is deleted.")

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		logger.info(
			f"Running passage filter node - {self.__class__.__name__} module..."
		)
		validate_qa_dataset(previous_result)

		# find queries columns
		assert (
			"query" in previous_result.columns
		), "previous_result must have query column."
		queries = previous_result["query"].tolist()

		# find contents_list columns
		assert (
			"retrieved_contents" in previous_result.columns
		), "previous_result must have retrieved_contents column."
		contents = previous_result["retrieved_contents"].tolist()

		# find scores columns
		assert (
			"retrieve_scores" in previous_result.columns
		), "previous_result must have retrieve_scores column."
		scores = previous_result["retrieve_scores"].tolist()

		# find ids columns
		assert (
			"retrieved_ids" in previous_result.columns
		), "previous_result must have retrieved_ids column."
		ids = previous_result["retrieved_ids"].tolist()
		return queries, contents, scores, ids
