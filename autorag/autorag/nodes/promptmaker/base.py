import logging
from abc import ABCMeta
from pathlib import Path
from typing import Union

import pandas as pd

from autorag.schema.base import BaseModule
from autorag.utils.cast import cast_retrieve_infos

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

		query = previous_result["query"].tolist()
		retrieve_infos = cast_retrieve_infos(previous_result)
		prompt = kwargs.pop("prompt")
		return query, retrieve_infos["retrieved_contents"], prompt
