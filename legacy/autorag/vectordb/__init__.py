import os
from typing import List

from autorag.support import dynamically_find_function
from autorag.utils.util import load_yaml_config
from autorag.vectordb.base import BaseVectorStore


def get_support_vectordb(vectordb_name: str):
	support_vectordb = {
		"chroma": ("autorag.vectordb.chroma", "Chroma"),
		"Chroma": ("autorag.vectordb.chroma", "Chroma"),
		"milvus": ("autorag.vectordb.milvus", "Milvus"),
		"Milvus": ("autorag.vectordb.milvus", "Milvus"),
		"weaviate": ("autorag.vectordb.weaviate", "Weaviate"),
		"Weaviate": ("autorag.vectordb.weaviate", "Weaviate"),
		"pinecone": ("autorag.vectordb.pinecone", "Pinecone"),
		"Pinecone": ("autorag.vectordb.pinecone", "Pinecone"),
		"couchbase": ("autorag.vectordb.couchbase", "Couchbase"),
		"Couchbase": ("autorag.vectordb.couchbase", "Couchbase"),
		"qdrant": ("autorag.vectordb.qdrant", "Qdrant"),
		"Qdrant": ("autorag.vectordb.qdrant", "Qdrant"),
	}
	return dynamically_find_function(vectordb_name, support_vectordb)


def load_vectordb(vectordb_name: str, **kwargs):
	vectordb = get_support_vectordb(vectordb_name)
	return vectordb(**kwargs)


def load_vectordb_from_yaml(yaml_path: str, vectordb_name: str, project_dir: str):
	config_dict = load_yaml_config(yaml_path)
	vectordb_list = config_dict.get("vectordb", [])
	if len(vectordb_list) == 0 or vectordb_name == "default":
		chroma_path = os.path.join(project_dir, "resources", "chroma")
		return load_vectordb(
			"chroma",
			client_type="persistent",
			embedding_model="openai",
			collection_name="openai",
			path=chroma_path,
		)

	target_dict = list(filter(lambda x: x["name"] == vectordb_name, vectordb_list))
	target_dict[0].pop("name")  # delete a name key
	target_vectordb_name = target_dict[0].pop("db_type")
	target_vectordb_params = target_dict[0]
	return load_vectordb(target_vectordb_name, **target_vectordb_params)


def load_all_vectordb_from_yaml(
	yaml_path: str, project_dir: str
) -> List[BaseVectorStore]:
	config_dict = load_yaml_config(yaml_path)
	vectordb_list = config_dict.get("vectordb", [])
	if len(vectordb_list) == 0:
		chroma_path = os.path.join(project_dir, "resources", "chroma")
		return [
			load_vectordb(
				"chroma",
				client_type="persistent",
				embedding_model="openai",
				collection_name="openai",
				path=chroma_path,
			)
		]

	result_vectordbs = []
	for vectordb_dict in vectordb_list:
		_ = vectordb_dict.pop("name")
		vectordb_type = vectordb_dict.pop("db_type")
		vectordb = load_vectordb(vectordb_type, **vectordb_dict)
		result_vectordbs.append(vectordb)
	return result_vectordbs
