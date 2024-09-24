import pandas as pd

from autorag.nodes.queryexpansion.base import BaseQueryExpansion
from autorag.utils import result_to_dataframe


class PassQueryExpansion(BaseQueryExpansion):
	@result_to_dataframe(["queries"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		"""
		Do not perform query expansion.
		Return with the same queries.
		The dimension will be 2-d list, and the column name will be 'queries'.
		"""
		assert (
			"query" in previous_result.columns
		), "previous_result must have query column."
		queries = previous_result["query"].tolist()
		return list(map(lambda x: [x], queries))

	def _pure(self, *args, **kwargs):
		pass
