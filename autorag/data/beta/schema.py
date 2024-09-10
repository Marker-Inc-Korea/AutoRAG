import logging
from typing import Callable, Optional, Dict, Awaitable, Any
import pandas as pd

from autorag.support import get_support_modules
from autorag.utils.util import process_batch, get_event_loop, fetch_contents

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
		self, fn: Callable[[Dict, Any], Awaitable[Dict]], batch_size: int = 32, **kwargs
	) -> "Raw":
		raw_dicts = self.data.to_dict(orient="records")
		loop = get_event_loop()
		tasks = [fn(raw_dict, **kwargs) for raw_dict in raw_dicts]
		results = loop.run_until_complete(process_batch(tasks, batch_size))
		return Raw(pd.DataFrame(results))

	def map(self, fn: Callable[[pd.DataFrame, Any], pd.DataFrame], **kwargs) -> "Raw":
		return Raw(fn(self.data, **kwargs))

	def flatmap(self, fn: Callable, **kwargs) -> "Raw":
		return fn(self.data, **kwargs)

	def chunk(self, module_name: str, **module_params) -> "Corpus":
		chunk_module = get_support_modules(module_name)
		chunked_result = chunk_module(parsed_result=self.data, **module_params)
		return Corpus(chunked_result, self)

	def __add__(self, other):
		assert isinstance(other, Raw), "You can only add Raw instances."
		self.data = pd.concat([self.data, other.data], ignore_index=True).reset_index(
			drop=True
		)
		return self


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
		self._linked_raw = linked_raw

	@property
	def linked_raw(self) -> Raw:
		return self._linked_raw

	@linked_raw.setter
	def linked_raw(self, raw: Raw):
		raise NotImplementedError("linked_raw is read-only.")

	def batch_apply(
		self, fn: Callable[[Dict, Any], Awaitable[Dict]], batch_size: int = 32, **kwargs
	) -> "Corpus":
		corpus_dicts = self.data.to_dict(orient="records")
		loop = get_event_loop()
		tasks = [fn(corpus_dict, **kwargs) for corpus_dict in corpus_dicts]
		results = loop.run_until_complete(process_batch(tasks, batch_size))
		return Corpus(pd.DataFrame(results), self.linked_raw)

	def map(
		self, fn: Callable[[pd.DataFrame, Any], pd.DataFrame], **kwargs
	) -> "Corpus":
		return Corpus(fn(self.data, **kwargs), self.linked_raw)

	def sample(self, fn: Callable[[pd.DataFrame, Any], pd.DataFrame], **kwargs) -> "QA":
		"""
		Sample the corpus for making QA.
		It selects the subset of the corpus and makes QA set from it.
		You can generate questions from the created question.
		It is the first step to make QA set from the corpus.
		If you select just one passage from each passage, it will be a single-hop QA set.
		If you select multiple passages from each passage, it will be a multi-hop QA set.

		:param fn: The select function to perform.
		It returns QA dataframe.
		:return: QA instance that is selected.
		It contains qid and retrieval_gt columns.
		"""
		return QA(fn(self.data, **kwargs), self)


class QA:
	def __init__(
		self,
		qa_df: Optional[pd.DataFrame] = None,
		linked_corpus: Optional[Corpus] = None,
	):
		self.data = qa_df
		self._linked_corpus = linked_corpus

	@property
	def linked_corpus(self) -> Corpus:
		return self._linked_corpus

	@linked_corpus.setter
	def linked_corpus(self, corpus: Corpus):
		raise NotImplementedError("linked_corpus is read-only.")

	def batch_apply(
		self, fn: Callable[[Dict, Any], Awaitable[Dict]], batch_size: int = 32, **kwargs
	) -> "QA":
		qa_dicts = self.data.to_dict(orient="records")
		loop = get_event_loop()
		tasks = [fn(qa_dict, **kwargs) for qa_dict in qa_dicts]
		results = loop.run_until_complete(process_batch(tasks, batch_size))
		return QA(pd.DataFrame(results), self.linked_corpus)

	def batch_filter(
		self, fn: Callable[[Dict, Any], Awaitable[bool]], batch_size: int = 32, **kwargs
	) -> "QA":
		qa_dicts = self.data.to_dict(orient="records")
		loop = get_event_loop()
		tasks = [fn(qa_dict, **kwargs) for qa_dict in qa_dicts]
		masks = loop.run_until_complete(process_batch(tasks, batch_size))
		return QA(self.data[masks], self.linked_corpus)

	def filter(self, fn: Callable[[Dict, Any], bool], **kwargs) -> "QA":
		qa_dicts = self.data.to_dict(orient="records")
		masks = [fn(qa_dict, **kwargs) for qa_dict in qa_dicts]
		return QA(self.data[masks], self.linked_corpus)

	def map(self, fn: Callable[[pd.DataFrame, Any], pd.DataFrame], **kwargs) -> "QA":
		return QA(fn(self.data, **kwargs), self.linked_corpus)

	def make_retrieval_gt_contents(self) -> "QA":
		"""
		Make retrieval_gt_contents column from retrieval_gt column.
		:return: The QA instance that has a retrieval_gt_contents column.
		"""
		self.data["retrieval_gt_contents"] = self.data["retrieval_gt"].apply(
			lambda x: fetch_contents(self.linked_corpus.data, x)
		)
		return self

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
