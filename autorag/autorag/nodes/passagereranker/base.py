import abc
import logging
from pathlib import Path
from typing import Union

import pandas as pd

from autorag.schema import BaseModule
from autorag.utils import validate_qa_dataset
from autorag.utils.cast import cast_retrieve_infos

logger = logging.getLogger("AutoRAG")


class BasePassageReranker(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs):
		logger.info(
			f"Initialize passage reranker node - {self.__class__.__name__} module..."
		)

	def __del__(self):
		logger.info(
			f"Deleting passage reranker node - {self.__class__.__name__} module..."
		)

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		logger.info(
			f"Running passage reranker node - {self.__class__.__name__} module..."
		)
		validate_qa_dataset(previous_result)

		# find queries columns
		assert (
			"query" in previous_result.columns
		), "previous_result must have query column."
		queries = previous_result["query"].tolist()

		retrieve_infos = cast_retrieve_infos(previous_result)
		return (
			queries,
			retrieve_infos["retrieved_contents"],
			retrieve_infos["retrieve_scores"],
			retrieve_infos["retrieved_ids"],
		)
