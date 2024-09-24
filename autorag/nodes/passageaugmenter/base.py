import abc
import logging
import os

import pandas as pd

from autorag.schema import BaseModule
from autorag.utils import (
	validate_qa_dataset,
	sort_by_scores,
	validate_corpus_dataset,
	cast_corpus_dataset,
)
from autorag.utils.util import select_top_k

logger = logging.getLogger("AutoRAG")


class BasePassageAugmenter(BaseModule, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: str, *args, **kwargs):
		logger.info(
			f"Initialize passage augmenter node - {self.__class__.__name__} module..."
		)
		data_dir = os.path.join(project_dir, "data")
		corpus_df = pd.read_parquet(
			os.path.join(data_dir, "corpus.parquet"), engine="pyarrow"
		)
		validate_corpus_dataset(corpus_df)
		corpus_df = cast_corpus_dataset(corpus_df)
		self.corpus_df = corpus_df

	def __del__(self):
		logger.info(
			f"Initialize passage augmenter node - {self.__class__.__name__} module..."
		)

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		logger.info(
			f"Running passage augmenter node - {self.__class__.__name__} module..."
		)
		validate_qa_dataset(previous_result)

		# find ids columns
		assert (
			"retrieved_ids" in previous_result.columns
		), "previous_result must have retrieved_ids column."
		ids = previous_result["retrieved_ids"].tolist()

		return ids

	@staticmethod
	def sort_by_scores(
		augmented_contents,
		augmented_ids,
		augmented_scores,
		top_k: int,
		reverse: bool = True,
	):
		# sort by scores
		df = pd.DataFrame(
			{
				"contents": augmented_contents,
				"ids": augmented_ids,
				"scores": augmented_scores,
			}
		)
		df[["contents", "ids", "scores"]] = df.apply(
			lambda row: sort_by_scores(row, reverse=reverse),
			axis=1,
			result_type="expand",
		)

		# select by top_k
		results = select_top_k(df, ["contents", "ids", "scores"], top_k)

		return (
			results["contents"].tolist(),
			results["ids"].tolist(),
			results["scores"].tolist(),
		)
