from typing import List, Union

import numpy as np
import pandas as pd

from autorag.embedding.base import EmbeddingModel
from autorag.evaluation.metric.util import calculate_cosine_similarity
from autorag.nodes.passageaugmenter.base import BasePassageAugmenter
from autorag.utils.util import (
	filter_dict_keys,
	fetch_contents,
	embedding_query_content,
	result_to_dataframe,
	empty_cuda_cache,
)


class PrevNextPassageAugmenter(BasePassageAugmenter):
	def __init__(
		self,
		project_dir: str,
		embedding_model: Union[str, dict] = "openai",
		*args,
		**kwargs,
	):
		"""
		Initialize the PrevNextPassageAugmenter module.

		:param project_dir:
		:param embedding_model: The embedding model name to use for calculating cosine similarity
			Default is openai (text-embedding-ada-002)
		:param kwargs:
		"""
		super().__init__(project_dir, *args, **kwargs)
		slim_corpus_df = self.corpus_df[["doc_id", "metadata"]]
		slim_corpus_df.loc[:, "metadata"] = slim_corpus_df["metadata"].apply(
			filter_dict_keys, keys=["prev_id", "next_id"]
		)
		self.slim_corpus_df = slim_corpus_df

		# init embedding model
		self.embedding_model = EmbeddingModel.load(embedding_model)()

	def __del__(self):
		del self.embedding_model
		empty_cuda_cache()
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		"""
		Run the passage augmenter node - PrevNextPassageAugmenter module.

		:param previous_result: The previous result Dataframe.
		:param top_k: You must input the top_k value to get the top k results.
		:param kwargs: Not affected.
		:return: DataFrame with retrieved_contents, retrieved_ids, and retrieve_scores columns
		"""
		top_k = kwargs.pop("top_k")

		ids = self.cast_to_run(previous_result)
		# find queries columns
		assert (
			"query" in previous_result.columns
		), "previous_result must have query column."
		queries = previous_result["query"].tolist()

		mode = kwargs.pop("mode", "both")
		num_passages = kwargs.pop("num_passages", 1)
		augmented_ids = self._pure(ids, num_passages, mode)

		# fetch contents from corpus to use augmented ids
		augmented_contents = fetch_contents(self.corpus_df, augmented_ids)

		query_embeddings, contents_embeddings = embedding_query_content(
			queries, augmented_contents, self.embedding_model, batch=128
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
		return self.sort_by_scores(
			augmented_contents, augmented_ids, augmented_scores, top_k
		)

	def _pure(
		self,
		ids_list: List[List[str]],
		num_passages: int = 1,
		mode: str = "both",
	) -> List[List[str]]:
		"""
		Add passages before and/or after the retrieved passage.
		For more information, visit https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/PrevNextPostprocessorDemo/.

		:param ids_list: The list of lists of ids retrieved
		:param num_passages: The number of passages to add before and after the retrieved passage
		    Default is 1.
		:param mode: The mode of augmentation
		    'prev': add passages before the retrieved passage
		    'next': add passages after the retrieved passage
		    'both': add passages before and after the retrieved passage
		    Default is 'next'.
		:return: The list of lists of augmented ids
		"""
		if mode not in ["prev", "next", "both"]:
			raise ValueError(f"mode must be 'prev', 'next', or 'both', but got {mode}")

		augmented_ids = [
			(
				lambda ids: prev_next_augmenter_pure(
					ids, self.slim_corpus_df, mode, num_passages
				)
			)(ids)
			for ids in ids_list
		]

		return augmented_ids


def prev_next_augmenter_pure(
	ids: List[str], corpus_df: pd.DataFrame, mode: str, num_passages: int
):
	def fetch_id_sequence(start_id, key):
		sequence = []
		current_id = start_id
		for _ in range(num_passages):
			current_id = (
				corpus_df.loc[corpus_df["doc_id"] == current_id]["metadata"]
				.values[0]
				.get(key)
			)
			if current_id is None:
				break
			sequence.append(current_id)
		return sequence

	augmented_group = []
	for id_ in ids:
		current_ids = [id_]
		if mode in ["prev", "both"]:
			current_ids = fetch_id_sequence(id_, "prev_id")[::-1] + current_ids
		if mode in ["next", "both"]:
			current_ids += fetch_id_sequence(id_, "next_id")
		augmented_group.extend(current_ids)
	return augmented_group
