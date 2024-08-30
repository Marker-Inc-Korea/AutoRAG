import logging
from typing import Callable, Optional, Dict, Awaitable
import pandas as pd
from autorag.utils.util import process_batch, get_event_loop

logger = logging.getLogger("AutoRAG")


class Corpus:
	def __init__(self, corpus_df: Optional[pd.DataFrame] = None):
		self.data = corpus_df

	def batch_apply(
		self, fn: Callable[[Dict], Awaitable[Dict]], batch_size: int = 32
	) -> "Corpus":
		corpus_dicts = self.data.to_dict(orient="records")
		loop = get_event_loop()
		tasks = [fn(corpus_dict) for corpus_dict in corpus_dicts]
		results = loop.run_until_complete(process_batch(tasks, batch_size))
		return Corpus(pd.DataFrame(results))

	def map(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> "Corpus":
		return Corpus(fn(self.data))


class QA:
	def __init__(self, qa_df: Optional[pd.DataFrame] = None):
		self.data = qa_df

	def batch_apply(
		self, fn: Callable[[Dict], Awaitable[Dict]], batch_size: int = 32
	) -> "QA":
		qa_dicts = self.data.to_dict(orient="records")
		loop = get_event_loop()
		tasks = [fn(qa_dict) for qa_dict in qa_dicts]
		results = loop.run_until_complete(process_batch(tasks, batch_size))
		return QA(pd.DataFrame(results))

	def map(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> "QA":
		return QA(fn(self.data))


class Dataset:
	def __init__(self, qa: QA = None, corpus: Corpus = None):
		self.qa = qa
		self.corpus = corpus

	def map(self, fn: Callable) -> "Dataset":
		qa_df, corpus_df = fn(self.qa, self.corpus)
		return Dataset(qa_df, corpus_df)

	def flatmap(self, fn: Callable) -> "Dataset":
		dataset = fn(self.qa, self.corpus)
		if not isinstance(dataset, Dataset):
			logger.warning(f"Expected Dataset, got {type(dataset)}")
			return Dataset(None, None)
		return dataset

	def qa_map(self, fn: Callable) -> "Dataset":
		qa_df = fn(self.qa)
		return Dataset(qa_df, self.corpus)

	def corpus_map(self, fn: Callable) -> "Dataset":
		corpus_df = fn(self.corpus)
		return Dataset(self.qa, corpus_df)
