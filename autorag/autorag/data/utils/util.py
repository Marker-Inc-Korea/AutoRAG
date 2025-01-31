import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import pandas as pd
import yaml
from langchain_core.documents import Document
from llama_index.core.schema import NodeRelationship

from autorag.schema import Module
from autorag.utils.util import make_combinations, explode


def get_file_metadata(file_path: str) -> Dict:
	"""Get some handy metadate from filesystem.

	Args:
	    file_path: str: file path in str
	"""
	return {
		"file_path": file_path,
		"file_name": os.path.basename(file_path),
		"file_type": mimetypes.guess_type(file_path)[0],
		"file_size": os.path.getsize(file_path),
		"creation_datetime": datetime.fromtimestamp(
			Path(file_path).stat().st_ctime
		).strftime("%Y-%m-%d"),
		"last_modified_datetime": datetime.fromtimestamp(
			Path(file_path).stat().st_mtime
		).strftime("%Y-%m-%d"),
		"last_accessed_datetime": datetime.fromtimestamp(
			Path(file_path).stat().st_atime
		).strftime("%Y-%m-%d"),
	}


def add_essential_metadata(metadata: Dict) -> Dict:
	if "last_modified_datetime" not in metadata:
		metadata["last_modified_datetime"] = datetime.now()
	return metadata


def corpus_df_to_langchain_documents(corpus_df: pd.DataFrame) -> List[Document]:
	page_contents = corpus_df["contents"].tolist()
	ids = corpus_df["doc_id"].tolist()
	metadatas = corpus_df["metadata"].tolist()
	return list(
		map(
			lambda x: Document(page_content=x[0], metadata={"filename": x[1], **x[2]}),
			zip(page_contents, ids, metadatas),
		)
	)


def add_essential_metadata_llama_text_node(metadata: Dict, relationships: Dict) -> Dict:
	if "last_modified_datetime" not in metadata:
		metadata["last_modified_datetime"] = datetime.now()

	if "prev_id" not in metadata:
		if NodeRelationship.PREVIOUS in relationships:
			prev_node = relationships.get(NodeRelationship.PREVIOUS, None)
			if prev_node:
				metadata["prev_id"] = prev_node.node_id

	if "next_id" not in metadata:
		if NodeRelationship.NEXT in relationships:
			next_node = relationships.get(NodeRelationship.NEXT, None)
			if next_node:
				metadata["next_id"] = next_node.node_id
	return metadata


def load_yaml(yaml_path: str):
	if not os.path.exists(yaml_path):
		raise ValueError(f"YAML file {yaml_path} does not exist.")
	with open(yaml_path, "r", encoding="utf-8") as stream:
		try:
			yaml_dict = yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			raise ValueError(f"YAML file {yaml_path} could not be loaded.") from exc
	return yaml_dict["modules"]


def get_param_combinations(modules: List[Dict]) -> Tuple[List[Callable], List[Dict]]:
	module_callable_list, module_params_list = [], []
	for module in modules:
		module_instance = Module.from_dict(module)
		module_params_list.append(module_instance.module_param)
		module_callable_list.append(module_instance.module)

	combinations = list(map(make_combinations, module_params_list))
	module_list, combination_list = explode(module_callable_list, combinations)
	return module_list, combination_list


def get_start_end_idx(original_text: str, search_str: str) -> Tuple[int, int]:
	start_idx = original_text.find(search_str)
	if start_idx == -1:
		return 0, 0
	end_idx = start_idx + len(search_str)
	return start_idx, end_idx - 1
