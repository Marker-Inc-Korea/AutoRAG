import functools
import logging
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd

from autorag import embedding_models
from autorag.data import chunk_modules, sentence_splitter_modules
from autorag.utils import result_to_dataframe

logger = logging.getLogger("AutoRAG")


def chunker_node(func):
	@functools.wraps(func)
	@result_to_dataframe(["doc_id", "contents", "metadata"])
	def wrapper(
		parsed_result: pd.DataFrame, chunk_method: Optional[str] = None, **kwargs
	) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
		logger.info(f"Running chunker - {func.__name__} module...")

		# get texts from parsed_result
		texts = parsed_result["texts"].tolist()

		# get filenames from parsed_result when 'add_file_name' is setting
		file_names_dict, kwargs = __make_file_names_dict(parsed_result, **kwargs)

		# run chunk module
		if func.__name__ in ["llama_index_chunk", "langchain_chunk"]:
			if chunk_method is None:
				raise ValueError(
					f"chunk_method is required for {func.__name__} module."
				)
			chunk_instance = __get_chunk_instance(
				func.__name__, chunk_method.lower(), **kwargs
			)
			result = func(texts=texts, chunker=chunk_instance, **file_names_dict)
			return result
		else:
			raise ValueError(f"Unsupported module_type: {func.__name__}")

	return wrapper


def __make_file_names_dict(parsed_result: pd.DataFrame, **kwargs):
	file_names_dict = {}

	if "add_file_name" in kwargs:
		file_name_language = kwargs.pop("add_file_name").lower()

		if "file_name" in parsed_result.columns:
			file_names = parsed_result["file_name"].tolist()
			file_names_dict = {
				"file_name_language": file_name_language,
				"file_names": file_names,
			}
		else:
			raise ValueError("The 'file_name' column is required in parsed_result")

	return file_names_dict, kwargs


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
			sentence_splitter = sentence_splitter_modules[sentence_splitter_str]
			kwargs.update({"sentence_splitter": sentence_splitter})

	def get_embedding_model(_embed_model_str: str, _module_type: str):
		if _embed_model_str == "openai":
			if _module_type == "langchain_chunk":
				_embed_model_str = "openai_langchain"
		return embedding_models[_embed_model_str]()

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
	if file_name_language == "english":
		return list(
			map(
				lambda x: f"file_name: {x[1]}\n contents: {x[0]}",
				zip(chunk_texts, file_names),
			)
		)
	elif file_name_language == "korean":
		return list(
			map(
				lambda x: f"파일 제목: {x[1]}\n 내용: {x[0]}",
				zip(chunk_texts, file_names),
			)
		)
	else:
		raise ValueError(
			f"Unsupported file_name_language: {file_name_language}. Choose from 'english' or 'korean'."
		)
