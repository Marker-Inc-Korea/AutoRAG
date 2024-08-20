import functools
import logging
import os
from pathlib import Path
from typing import List, Union

import pandas as pd

from autorag.utils import result_to_dataframe, fetch_contents

logger = logging.getLogger("AutoRAG")


def prompt_maker_node(func):
	@functools.wraps(func)
	@result_to_dataframe(["prompts"])
	def wrapper(
		project_dir: Union[str, Path], previous_result: pd.DataFrame, *args, **kwargs
	) -> List[str]:
		logger.info(f"Running prompt maker node - {func.__name__} module...")
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

		if func.__name__ == "fstring":
			return func(prompt, query, retrieved_contents)
		elif func.__name__ == "long_context_reorder":
			assert (
				"retrieve_scores" in previous_result.columns
			), "previous_result must have retrieve_scores column."
			retrieve_scores = previous_result["retrieve_scores"].tolist()
			return func(prompt, query, retrieved_contents, retrieve_scores)
		elif func.__name__ == "window_replacement":
			retrieved_ids = previous_result["retrieved_ids"].tolist()
			# load corpus
			data_dir = os.path.join(project_dir, "data")
			corpus_data = pd.read_parquet(
				os.path.join(data_dir, "corpus.parquet"), engine="pyarrow"
			)
			# get metadata from corpus
			retrieved_metadata = fetch_contents(
				corpus_data, retrieved_ids, column_name="metadata"
			)
			return func(prompt, query, retrieved_contents, retrieved_metadata)
		else:
			raise NotImplementedError(
				f"Module {func.__name__} is not implemented or not supported."
			)

	return wrapper
