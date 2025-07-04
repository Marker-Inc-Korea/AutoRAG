import abc

import pandas as pd

from autorag.nodes.retrieval.base import BaseRetrieval
from autorag.utils import result_to_dataframe
from autorag.utils.util import pop_params, fetch_contents


class HybridRetrieval(BaseRetrieval, metaclass=abc.ABCMeta):
	def __init__(self, project_dir: str, *args, **kwargs):
		super().__init__(project_dir)

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		previous_info = self.cast_to_run(previous_result, *args, **kwargs)
		_pure_params = pop_params(self._pure, kwargs)
		ids, scores = self._pure(previous_info, **_pure_params)
		contents = fetch_contents(self.corpus_df, ids)
		return contents, ids, scores

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		return hybrid_cast(previous_result)

	@classmethod
	def cast_to_run_class(cls, previous_result: pd.DataFrame):
		return hybrid_cast(previous_result)


def hybrid_cast(
	previous_result: pd.DataFrame,
):
	assert "query" in previous_result.columns, "previous_result must have query column."
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
