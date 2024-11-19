from datetime import datetime

import numpy as np
import pandas as pd

from autorag.utils.util import preprocess_text


def validate_qa_dataset(df: pd.DataFrame):
	columns = ["qid", "query", "retrieval_gt", "generation_gt"]
	assert set(columns).issubset(
		df.columns
	), f"df must have columns {columns}, but got {df.columns}"


def validate_corpus_dataset(df: pd.DataFrame):
	columns = ["doc_id", "contents", "metadata"]
	assert set(columns).issubset(
		df.columns
	), f"df must have columns {columns}, but got {df.columns}"


def cast_qa_dataset(df: pd.DataFrame):
	def cast_retrieval_gt(gt):
		if isinstance(gt, str):
			return [[gt]]
		elif isinstance(gt, list):
			if isinstance(gt[0], str):
				return [gt]
			elif isinstance(gt[0], list):
				return gt
			elif isinstance(gt[0], np.ndarray):
				return cast_retrieval_gt(list(map(lambda x: x.tolist(), gt)))
			else:
				raise ValueError(
					f"retrieval_gt must be str or list, but got {type(gt[0])}"
				)
		elif isinstance(gt, np.ndarray):
			return cast_retrieval_gt(gt.tolist())
		else:
			raise ValueError(f"retrieval_gt must be str or list, but got {type(gt)}")

	def cast_generation_gt(gt):
		if isinstance(gt, str):
			return [gt]
		elif isinstance(gt, list):
			return gt
		elif isinstance(gt, np.ndarray):
			return cast_generation_gt(gt.tolist())
		else:
			raise ValueError(f"generation_gt must be str or list, but got {type(gt)}")

	df = df.reset_index(drop=True)
	validate_qa_dataset(df)
	assert df["qid"].apply(lambda x: isinstance(x, str)).sum() == len(
		df
	), "qid must be string type."
	assert df["query"].apply(lambda x: isinstance(x, str)).sum() == len(
		df
	), "query must be string type."
	df["retrieval_gt"] = df["retrieval_gt"].apply(cast_retrieval_gt)
	df["generation_gt"] = df["generation_gt"].apply(cast_generation_gt)
	df["query"] = df["query"].apply(preprocess_text)
	df["generation_gt"] = df["generation_gt"].apply(
		lambda x: list(map(preprocess_text, x))
	)
	return df


def cast_corpus_dataset(df: pd.DataFrame):
	df = df.reset_index(drop=True)
	validate_corpus_dataset(df)

	# drop rows that have empty contents
	df = df[~df["contents"].apply(lambda x: x is None or x.isspace())]

	def make_datetime_metadata(x):
		if x is None or x == {}:
			return {"last_modified_datetime": datetime.now()}
		elif x.get("last_modified_datetime") is None:
			return {**x, "last_modified_datetime": datetime.now()}
		else:
			return x

	df["metadata"] = df["metadata"].apply(make_datetime_metadata)

	# check every metadata have a datetime key
	assert sum(
		df["metadata"].apply(lambda x: x.get("last_modified_datetime") is not None)
	) == len(df), "Every metadata must have a datetime key."

	def make_prev_next_id_metadata(x, id_type: str):
		if x is None or x == {}:
			return {id_type: None}
		elif x.get(id_type) is None:
			return {**x, id_type: None}
		else:
			return x

	df["metadata"] = df["metadata"].apply(
		lambda x: make_prev_next_id_metadata(x, "prev_id")
	)
	df["metadata"] = df["metadata"].apply(
		lambda x: make_prev_next_id_metadata(x, "next_id")
	)

	df["contents"] = df["contents"].apply(preprocess_text)

	def normalize_unicode_metadata(metadata: dict):
		result = {}
		for key, value in metadata.items():
			if isinstance(value, str):
				result[key] = preprocess_text(value)
			else:
				result[key] = value
		return result

	df["metadata"] = df["metadata"].apply(normalize_unicode_metadata)

	# check every metadata have a prev_id, next_id key
	assert all(
		"prev_id" in metadata for metadata in df["metadata"]
	), "Every metadata must have a prev_id key."
	assert all(
		"next_id" in metadata for metadata in df["metadata"]
	), "Every metadata must have a next_id key."

	return df


def validate_qa_from_corpus_dataset(qa_df: pd.DataFrame, corpus_df: pd.DataFrame):
	qa_ids = []
	for retrieval_gt in qa_df["retrieval_gt"].tolist():
		if isinstance(retrieval_gt, list) and (
			retrieval_gt[0] != [] or any(bool(g) is True for g in retrieval_gt)
		):
			for gt in retrieval_gt:
				qa_ids.extend(gt)
		elif isinstance(retrieval_gt, np.ndarray) and retrieval_gt[0].size > 0:
			for gt in retrieval_gt:
				qa_ids.extend(gt)

	no_exist_ids = list(
		filter(lambda qa_id: corpus_df[corpus_df["doc_id"] == qa_id].empty, qa_ids)
	)

	assert (
		len(no_exist_ids) == 0
	), f"{len(no_exist_ids)} doc_ids in retrieval_gt do not exist in corpus_df."
