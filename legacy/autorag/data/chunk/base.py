import functools
import logging
from typing import Tuple, List, Dict, Any

import pandas as pd

from autorag.embedding.base import EmbeddingModel
from autorag.data import chunk_modules, sentence_splitter_modules
from autorag.utils import result_to_dataframe

logger = logging.getLogger("AutoRAG")


def chunker_node(func):
	@functools.wraps(func)
	@result_to_dataframe(["doc_id", "contents", "path", "start_end_idx", "metadata"])
	def wrapper(
		parsed_result: pd.DataFrame, chunk_method: str, **kwargs
	) -> Tuple[
		List[str], List[str], List[str], List[Tuple[int, int]], List[Dict[str, Any]]
	]:
		logger.info(f"Running chunker - {func.__name__} module...")

		# get texts from parsed_result
		texts = parsed_result["texts"].tolist()

		# get filenames from parsed_result when 'add_file_name' is setting
		file_name_language = kwargs.pop("add_file_name", None)
		metadata_list = make_metadata_list(parsed_result)

		# run chunk module
		if func.__name__ in ["llama_index_chunk", "langchain_chunk"]:
			chunk_instance = __get_chunk_instance(
				func.__name__, chunk_method.lower(), **kwargs
			)
			result = func(
				texts=texts,
				chunker=chunk_instance,
				file_name_language=file_name_language,
				metadata_list=metadata_list,
			)
			del chunk_instance
			return result
		else:
			raise ValueError(f"Unsupported module_type: {func.__name__}")

	return wrapper


def make_metadata_list(parsed_result: pd.DataFrame) -> List[Dict[str, str]]:
	metadata_list = [{} for _ in range(len(parsed_result["texts"]))]

	def _make_metadata_pure(
		lst: List[str], key: str, metadata_lst: List[Dict[str, str]]
	):
		for value, metadata in zip(lst, metadata_lst):
			metadata[key] = value

	for column in ["page", "last_modified_datetime", "path"]:
		if column in parsed_result.columns:
			_make_metadata_pure(parsed_result[column].tolist(), column, metadata_list)
	return metadata_list


def __get_chunk_instance(module_type: str, chunk_method: str, **kwargs):
	# Add sentence_splitter to kwargs
	sentence_available_methods = [
		"semantic_llama_index",
		"semanticdoublemerging",
		"sentencewindow",
	]
	if chunk_method in sentence_available_methods:
		# llama index default sentence_splitter is 'nltk -PunktSentenceTokenizer'
		if "sentence_splitter" in kwargs.keys():
			sentence_splitter_str = kwargs.pop("sentence_splitter")
			sentence_splitter_func = sentence_splitter_modules[sentence_splitter_str]()
			kwargs.update({"sentence_splitter": sentence_splitter_func})

	def get_embedding_model(_embed_model_str: str, _module_type: str):
		if _embed_model_str == "openai":
			if _module_type == "langchain_chunk":
				_embed_model_str = "openai_langchain"
		return EmbeddingModel.load(_embed_model_str)()

	# Add embed_model to kwargs
	embedding_available_methods = ["semantic_llama_index", "semantic_langchain"]
	if chunk_method in embedding_available_methods:
		# there is no default embed_model, so we have to get it parameter and add it.
		if "embed_model" not in kwargs.keys():
			raise ValueError(f"embed_model is required for {chunk_method} method.")
		embed_model_str = kwargs.pop("embed_model")
		embed_model = get_embedding_model(embed_model_str, module_type)
		if chunk_method == "semantic_llama_index":
			kwargs.update({"embed_model": embed_model})
		elif chunk_method == "semantic_langchain":
			kwargs.update({"embeddings": embed_model})

	return chunk_modules[chunk_method](**kwargs)


def add_file_name(
	file_name_language: str, file_names: List[str], chunk_texts: List[str]
) -> List[str]:
	if file_name_language == "en":
		return list(
			map(
				lambda x: f"file_name: {x[1]}\n contents: {x[0]}",
				zip(chunk_texts, file_names),
			)
		)
	elif file_name_language == "ko":
		return list(
			map(
				lambda x: f"파일 제목: {x[1]}\n 내용: {x[0]}",
				zip(chunk_texts, file_names),
			)
		)
	elif file_name_language == "ja":
		return list(
			map(
				lambda x: f"ファイル名: {x[1]}\n 内容: {x[0]}",
				zip(chunk_texts, file_names),
			)
		)
	else:
		raise ValueError(
			f"Unsupported file_name_language: {file_name_language}. Choose from 'en' ,'ko' or 'ja."
		)
