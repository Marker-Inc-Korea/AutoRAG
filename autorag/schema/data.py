import logging
from typing import Callable, Optional, Dict, Awaitable
import pandas as pd
from autorag.utils.util import process_batch, get_event_loop

logger = logging.getLogger("AutoRAG")


class Raw:
	"""
	The Raw class that stored document parsing results.
	It can do chunking.
	It has two column names, 'raw_id' and 'contents'.
	"""

	def __init__(self, raw_df: Optional[pd.DataFrame] = None):
		self.data = raw_df

	def batch_apply(
		self, fn: Callable[[Dict], Awaitable[Dict]], batch_size: int = 32
	) -> "Raw":
		raw_dicts = self.data.to_dict(orient="records")
		loop = get_event_loop()
		tasks = [fn(raw_dict) for raw_dict in raw_dicts]
		results = loop.run_until_complete(process_batch(tasks, batch_size))
		return Raw(pd.DataFrame(results))

	def map(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> "Raw":
		return Raw(fn(self.data))

	def flatmap(self, fn: Callable) -> "Raw":
		return fn(self.data)

	def chunk(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> "Corpus":
		return Corpus(fn(self.data), self)


class Corpus:
	"""
	The Corpus class that stored chunked passages.
	It can generate qa set, linked with Raw instance.
	"""

	def __init__(
		self,
		corpus_df: Optional[pd.DataFrame] = None,
		linked_raw: Optional[Raw] = None,
	):
		self.data = corpus_df
		self.linked_raw = linked_raw

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

	def select_evidence(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> "Evidence":
		return Evidence(fn(self.data), self)


class Evidence:
	def __init__(
		self,
		evidence_df: Optional[pd.DataFrame] = None,
		linked_corpus: Optional[Corpus] = None,
	):
		self.data = evidence_df
		self.linked_corpus = linked_corpus

	def map(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> "Evidence":
		return Evidence(fn(self.data), self.linked_corpus)

	def generate_qa(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> "QA":
		return QA(fn(self.data), self.linked_corpus)


class QA:
	def __init__(
		self,
		qa_df: Optional[pd.DataFrame] = None,
		linked_corpus: Optional[Corpus] = None,
	):
		self.data = qa_df
		self.linked_corpus = linked_corpus

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

	def update_corpus(self, new_corpus: Corpus) -> "QA":
		"""
		Update linked corpus.
		Not just replace linked_corpus to the new Corpus,
		it replaces the whole `retrieval_gt` to the new corpus using `linked_raw`.
		The QA data must have a `retrieval_gt` column.

		:param new_corpus: Corpus that you want to replace.
			Must have valid `linked_raw` and `raw_id`, `raw_start_idx`, `raw_end_idx` columns.
		:return: The QA instance that updated linked corpus.
		"""
		pass
