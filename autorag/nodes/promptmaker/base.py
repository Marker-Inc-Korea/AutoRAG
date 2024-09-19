import logging
from abc import ABCMeta
from pathlib import Path
from typing import Union

import pandas as pd

from autorag.schema.base import BaseModule

logger = logging.getLogger("AutoRAG")


class BasePromptMaker(BaseModule, metaclass=ABCMeta):
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs):
		logger.info(
			f"Initialize prompt maker node - {self.__class__.__name__} module..."
		)

	def __del__(self):
		logger.info(f"Prompt maker node - {self.__class__.__name__} module is deleted.")

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		logger.info(f"Running prompt maker node - {self.__class__.__name__} module...")
		# get query and retrieved contents from previous_result
		assert (
			"query" in previous_result.columns
		), "previous_result must have query column."
		assert (
			"retrieved_contents" in previous_result.columns
		), "previous_result must have retrieved_contents column."
		query = previous_result["query"].tolist()
		retrieved_contents = previous_result["retrieved_contents"].tolist()
		prompt = kwargs.pop("prompt")
		return query, retrieved_contents, prompt
