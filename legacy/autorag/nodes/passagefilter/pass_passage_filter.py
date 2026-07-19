import pandas as pd

from autorag.nodes.passagefilter.base import BasePassageFilter
from autorag.utils import result_to_dataframe


class PassPassageFilter(BasePassageFilter):
	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		_, contents, scores, ids = self.cast_to_run(previous_result)
		return contents, ids, scores

	def _pure(self, *args, **kwargs):
		pass
