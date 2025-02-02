import logging
from typing import Callable, Optional, Dict, Awaitable, Any, Tuple, List
import uuid
import pandas as pd
from autorag.utils.util import process_batch, get_event_loop, fetch_contents

from autorag.support import get_support_modules

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

	def to_parquet(self, save_path: str):
		"""
		Save the corpus to the AutoRAG compatible parquet file.
		It is not for the data creation, for running AutoRAG.
		If you want to save it directly, use the below code.
		`corpus.data.to_parquet(save_path)`

		:param save_path: The path to save the corpus.
		"""
		if not save_path.endswith(".parquet"):
			raise ValueError("save_path must be ended with .parquet")
		save_df = self.data.reset_index(drop=True)
		save_df.to_parquet(save_path)

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

		# Experimental feature
		if fn.__name__ == "multiple_queries_gen":
			return self._process_multiple_queries_gen(results)

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

	def to_parquet(self, qa_save_path: str, corpus_save_path: str):
		"""
		Save the qa and corpus to the AutoRAG compatible parquet file.
		It is not for the data creation, for running AutoRAG.
		If you want to save it directly, use the below code.
		`qa.data.to_parquet(save_path)`

		:param qa_save_path: The path to save the qa dataset.
		:param corpus_save_path: The path to save the corpus.
		"""
		if not qa_save_path.endswith(".parquet"):
			raise ValueError("save_path must be ended with .parquet")
		if not corpus_save_path.endswith(".parquet"):
			raise ValueError("save_path must be ended with .parquet")
		save_df = self.data[
			["qid", "query", "retrieval_gt", "generation_gt"]
		].reset_index(drop=True)
		save_df.to_parquet(qa_save_path)
		self.linked_corpus.to_parquet(corpus_save_path)

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
		self.data["evidence_path"] = (
			self.data["retrieval_gt"]
			.apply(
				lambda x: fetch_contents(
					self.linked_corpus.data,
					x,
					column_name="path",
				)
			)
			.tolist()
		)
		self.data["evidence_page"] = self.data["retrieval_gt"].apply(
			lambda x: list(
				map(
					lambda lst: list(map(lambda x: x.get("page", -1), lst)),
					fetch_contents(self.linked_corpus.data, x, column_name="metadata"),
				)
			)
		)
		if "evidence_start_end_idx" not in self.data.columns:
			# make evidence start_end_idx
			self.data["evidence_start_end_idx"] = (
				self.data["retrieval_gt"]
				.apply(
					lambda x: fetch_contents(
						self.linked_corpus.data,
						x,
						column_name="start_end_idx",
					)
				)
				.tolist()
			)

		# matching the new corpus with the old corpus
		path_corpus_dict = QA.__make_path_corpus_dict(new_corpus.data)
		new_retrieval_gt = self.data.apply(
			lambda row: QA.__match_index_row(
				row["evidence_start_end_idx"],
				row["evidence_path"],
				row["evidence_page"],
				path_corpus_dict,
			),
			axis=1,
		).tolist()
		new_qa = self.data.copy(deep=True)[["qid", "query", "generation_gt"]]
		new_qa["retrieval_gt"] = new_retrieval_gt
		return QA(new_qa, new_corpus)

	@staticmethod
	def __match_index(target_idx: Tuple[int, int], dst_idx: Tuple[int, int]) -> bool:
		"""
		Check if the target_idx is overlap by the dst_idx.
		"""
		target_start, target_end = target_idx
		dst_start, dst_end = dst_idx
		return (
			dst_start <= target_start <= dst_end or dst_start <= target_end <= dst_end
		)

	@staticmethod
	def __match_index_row(
		evidence_indices: List[List[Tuple[int, int]]],
		evidence_paths: List[List[str]],
		evidence_pages: List[List[int]],
		path_corpus_dict: Dict,
	) -> List[List[str]]:
		"""
		Find the matched passage from new_corpus.

		:param evidence_indices: The evidence indices at the corresponding Raw.
		        Its shape is the same as the retrieval_gt.
		:param evidence_paths: The evidence paths at the corresponding Raw.
		        Its shape is the same as the retrieval_gt.
		:param path_corpus_dict: The key is the path name, and the value is the corpus dataframe that only contains the path in the key.
		        You can make it using `QA.__make_path_corpus_dict`.
		:return:
		"""
		result = []
		for i, idx_list in enumerate(evidence_indices):
			sub_result = []
			for j, idx in enumerate(idx_list):
				path_corpus_df = path_corpus_dict[evidence_paths[i][j]]
				if evidence_pages[i][j] >= 0:
					path_corpus_df = path_corpus_df.loc[
						path_corpus_df["metadata"].apply(lambda x: x.get("page", -1))
						== evidence_pages[i][j]
					]
				matched_corpus = path_corpus_df.loc[
					path_corpus_df["start_end_idx"].apply(
						lambda x: QA.__match_index(idx, x)
					)
				]
				sub_result.extend(matched_corpus["doc_id"].tolist())
			result.append(sub_result)
		return result

	@staticmethod
	def __make_path_corpus_dict(corpus_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
		return {
			path: corpus_df[corpus_df["path"] == path]
			for path in corpus_df["path"].unique()
		}

	# Experimental feature
	def _process_multiple_queries_gen(self, results: List[Dict]) -> "QA":
		data = []
		for result in results:
			queries = result["query"].split("\n")
			for query in queries:
				new_result = {
					key: (str(uuid.uuid4()) if key == "qid" else result[key])
					for key in result.keys()
				}
				new_result["query"] = query
				data.append(new_result)
		df = pd.DataFrame(data)
		return QA(df, self.linked_corpus)
