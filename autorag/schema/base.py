from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd


class BaseModule(metaclass=ABCMeta):
	@abstractmethod
	def __init__(self, project_dir: Union[str, Path], *args, **kwargs):
		pass

	@abstractmethod
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
		pass

	@abstractmethod
	def _pure(self, *args, **kwargs):
		pass

	@classmethod
	def run_evaluator(cls, *args, **kwargs):
		instance = cls(*args, **kwargs)
		result = instance.pure(*args, **kwargs)
		del instance
		return result

	@abstractmethod
	def cast_to_init(self, project_dir: Union[str, Path], *args, **kwargs):
		"""
		This function is for cast function (a.k.a decorator) only for init function in the whole node.
		"""
		pass

	@abstractmethod
	def cast_to_run(self, previous_result: pd.DataFrame, *args, **kwargs):
		"""
		This function is for cast function (a.k.a decorator) only for pure function in the whole node.
		"""
		pass
