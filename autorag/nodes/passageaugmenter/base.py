import functools
import logging
import os
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
import pandas as pd

from autorag import embedding_models
from autorag.evaluation.metric.util import calculate_cosine_similarity
from autorag.utils import (
	result_to_dataframe,
	validate_qa_dataset,
	fetch_contents,
	sort_by_scores,
	validate_corpus_dataset,
	cast_corpus_dataset,
)
from autorag.utils.util import filter_dict_keys, select_top_k, embedding_query_content

logger = logging.getLogger("AutoRAG")


def passage_augmenter_node(func):
	@functools.wraps(func)
	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def wrapper(
		project_dir: Union[str, Path], previous_result: pd.DataFrame, *args, **kwargs
	) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
		logger.info(f"Running passage augmenter node - {func.__name__} module...")
		validate_qa_dataset(previous_result)
		data_dir = os.path.join(project_dir, "data")

		# find queries columns
		assert (
			"query" in previous_result.columns
		), "previous_result must have query column."
		queries = previous_result["query"].tolist()

		# find ids columns
		assert (
			"retrieved_ids" in previous_result.columns
		), "previous_result must have retrieved_ids column."
		ids = previous_result["retrieved_ids"].tolist()

		corpus_df = pd.read_parquet(
			os.path.join(data_dir, "corpus.parquet"), engine="pyarrow"
		)
		validate_corpus_dataset(corpus_df)

		# get top_k
		top_k = kwargs.pop("top_k")

		if func.__name__ == "prev_next_augmenter":
			corpus_df = cast_corpus_dataset(corpus_df)
			slim_corpus_df = corpus_df[["doc_id", "metadata"]]
			slim_corpus_df.loc[:, "metadata"] = slim_corpus_df["metadata"].apply(
				filter_dict_keys, keys=["prev_id", "next_id"]
			)

			mode = kwargs.pop("mode", "both")
			num_passages = kwargs.pop("num_passages", 1)

			# get augmented ids
			augmented_ids = func(
				ids_list=ids,
				corpus_df=slim_corpus_df,
				mode=mode,
				num_passages=num_passages,
			)

			# fetch contents from corpus to use augmented ids
			augmented_contents = fetch_contents(corpus_df, augmented_ids)

			# set embedding model for getting scores
			embedding_model_str = kwargs.pop("embedding_model", "openai")
			embedding_model = embedding_models[embedding_model_str]
			query_embeddings, contents_embeddings = embedding_query_content(
				queries, augmented_contents, embedding_model, batch=128
			)

			# get scores from calculated cosine similarity
			augmented_scores = [
				np.array(
					[
						calculate_cosine_similarity(query_embedding, x)
						for x in content_embeddings
					]
				).tolist()
				for query_embedding, content_embeddings in zip(
					query_embeddings, contents_embeddings
				)
			]
		else:
			contents = previous_result["retrieved_contents"].tolist()
			scores = previous_result["retrieve_scores"].tolist()

			augmented_ids, augmented_contents, augmented_scores = func(
				ids_list=ids, contents_list=contents, scores_list=scores, **kwargs
			)

		# sort by scores
		df = pd.DataFrame(
			{
				"contents": augmented_contents,
				"ids": augmented_ids,
				"scores": augmented_scores,
			}
		)
		df[["contents", "ids", "scores"]] = df.apply(
			sort_by_scores, axis=1, result_type="expand"
		)

		# select by top_k
		results = select_top_k(df, ["contents", "ids", "scores"], top_k)

		return (
			results["contents"].tolist(),
			results["ids"].tolist(),
			results["scores"].tolist(),
		)

	return wrapper
