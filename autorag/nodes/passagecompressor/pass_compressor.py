from typing import List

import pandas as pd

from autorag.nodes.passagecompressor.base import BasePassageCompressor
from autorag.utils import result_to_dataframe


class PassCompressor(BasePassageCompressor):
	@result_to_dataframe(["retrieved_contents"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		_, contents = self.cast_to_run(previous_result)
		return self._pure(contents)

	def _pure(self, contents: List[List[str]]):
		return contents
