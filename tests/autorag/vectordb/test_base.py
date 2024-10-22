import os.path
import pathlib
import tempfile

from llama_index.core import MockEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.vectordb import (
	load_vectordb,
	load_vectordb_from_yaml,
	load_all_vectordb_from_yaml,
)
from autorag.vectordb.chroma import Chroma


root_path = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
resource_dir = os.path.join(root_path, "resources")


def test_load_vectordb():
	db = load_vectordb(
		"chroma",
		client_type="ephemeral",
		collection_name="jax1",
		embedding_model="mock",
	)
	assert isinstance(db, Chroma)
	assert db.collection.name == "jax1"


def test_load_vectordb_from_yaml():
	yaml_path = os.path.join(resource_dir, "simple_mock.yaml")
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		os.environ["PROJECT_DIR"] = project_dir
		default_vectordb = load_vectordb_from_yaml(yaml_path, "default", project_dir)
		assert isinstance(default_vectordb, Chroma)
		assert default_vectordb.collection.name == "openai"
		assert isinstance(default_vectordb.embedding, OpenAIEmbedding)

		chroma_default_vectordb = load_vectordb_from_yaml(
			yaml_path, "chroma_default", project_dir
		)
		assert isinstance(chroma_default_vectordb, Chroma)
		assert chroma_default_vectordb.collection.name == "openai"
		assert isinstance(chroma_default_vectordb.embedding, MockEmbedding)

		chroma_large_vectordb = load_vectordb_from_yaml(
			yaml_path, "chroma_large", project_dir
		)
		assert isinstance(chroma_large_vectordb, Chroma)
		assert chroma_large_vectordb.collection.name == "openai_embed_3_large"
		assert isinstance(chroma_large_vectordb.embedding, MockEmbedding)


def test_load_all_vectordb_from_yaml():
	yaml_path = os.path.join(resource_dir, "simple_mock.yaml")
	with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as project_dir:
		os.environ["PROJECT_DIR"] = project_dir
		vectordb_list = load_all_vectordb_from_yaml(yaml_path, project_dir)
		assert len(vectordb_list) == 2

		chroma_default_vectordb = vectordb_list[0]
		assert isinstance(chroma_default_vectordb, Chroma)
		assert chroma_default_vectordb.collection.name == "openai"
		assert isinstance(chroma_default_vectordb.embedding, MockEmbedding)

		chroma_large_vectordb = vectordb_list[1]
		assert isinstance(chroma_large_vectordb, Chroma)
		assert chroma_large_vectordb.collection.name == "openai_embed_3_large"
		assert isinstance(chroma_large_vectordb.embedding, MockEmbedding)
