from pathlib import Path
from typing import Union

import pandas as pd

from autorag.schema import BaseModule


class TestModule(BaseModule):
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs):
		self.param1 = kwargs.pop("param1", None)
		self.param2 = self.cast_to_init(project_dir, *args, **kwargs)

	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		param3 = self.cast_to_run(previous_result, *args, **kwargs)
		return param3, self._pure(*args, **kwargs)

	def _pure(self, *args, **kwargs):
		return [
			kwargs.pop("param1", None),
			kwargs.pop("param2", None),
			kwargs.pop("param3", None),
			kwargs,
		]

	def cast_to_init(self, project_dir: Union[str, Path], *args, **kwargs):
		return kwargs.pop("param2", None)

	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		return kwargs.pop("param3", None)


def test_base_module_kwargs_pop():
	param3, result_lst = TestModule.run_evaluator(
		"pseudo", previous_result=pd.DataFrame(), param1=1, param2=2, param3=3, param4=4
	)
	assert param3 == 3
	assert result_lst == [1, 2, 3, {"param4": 4}]
