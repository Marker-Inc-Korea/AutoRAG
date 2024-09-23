from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd


class BaseModule(metaclass=ABCMeta):
	@abstractmethod
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		pass

	@abstractmethod
	def _pure(self, *args, **kwargs):
		pass

	@classmethod
	def run_evaluator(
		cls,
		project_dir: Union[str, Path],
		previous_result: pd.DataFrame,
		*args,
		**kwargs,
	):
		instance = cls(project_dir, *args, **kwargs)
		result = instance.pure(previous_result, *args, **kwargs)
		del instance
		return result

	@abstractmethod
	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		"""
		This function is for cast function (a.k.a decorator) only for pure function in the whole node.
		"""
		pass
