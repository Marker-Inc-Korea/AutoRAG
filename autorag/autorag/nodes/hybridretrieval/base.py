import abc
import logging
from pathlib import Path
from typing import Union

import pandas as pd

from autorag.schema import BaseModule

logger = logging.getLogger("AutoRAG")


class BaseHybridRetrieval(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs) -> None:
		logger.info(f"Initialize hybrid retrieval node - {self.__class__.__name__}")

	def __del__(self) -> None:
		logger.info(
			f"Hybrid retrieval node - {self.__class__.__name__} module is deleted."
		)

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		# Cast the previous result to the adequate format for the run method.

		assert (
			"query" in previous_result.columns
		), "previous_result must have query column."
		queries = previous_result["query"].tolist()

		assert "retrieved_contents_semantic" in previous_result.columns
		assert "retrieved_contents_lexical" in previous_result.columns
		assert "retrieve_scores_semantic" in previous_result.columns
		assert "retrieve_scores_lexical" in previous_result.columns
		assert "retrieved_ids_semantic" in previous_result.columns
		assert "retrieved_ids_lexical" in previous_result.columns

		contents_semantic = previous_result["retrieved_contents_semantic"].tolist()
		contents_lexical = previous_result["retrieved_contents_lexical"].tolist()
		scores_semantic = previous_result["retrieve_scores_semantic"].tolist()
		scores_lexical = previous_result["retrieve_scores_lexical"].tolist()
		ids_semantic = previous_result["retrieved_ids_semantic"].tolist()
		ids_lexical = previous_result["retrieved_ids_lexical"].tolist()

		return {
			"queries": queries,
			"retrieved_contents_semantic": contents_semantic,
			"retrieved_contents_lexical": contents_lexical,
			"retrieve_scores_semantic": scores_semantic,
			"retrieve_scores_lexical": scores_lexical,
			"retrieved_ids_semantic": ids_semantic,
			"retrieved_ids_lexical": ids_lexical,
		}
