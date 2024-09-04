import pytest

from autorag.data import chunk_modules
from autorag.data.chunk import llama_index_chunk

from tests.autorag.data.chunk.test_chunk_base import base_texts, parsed_result


@pytest.fixture
def chunk_instance():
	chunk_method = "token"
	chunk_instance = chunk_modules[chunk_method]
	yield chunk_instance


def test_llama_index_chunk(chunk_instance):
	llama_index_chunk_original = llama_index_chunk.__wrapped__
	doc_id, contents, metadata = llama_index_chunk_original(base_texts, chunk_instance)
	assert contents


def test_llama_index_chunk_node():
	# result_df = llama_index_chunk(parsed_result, chunk_method="token")
	pass


def test_llama_index_chunk_file_name_ko():
	pass


def test_llama_index_chunk_file_name_ko_node():
	pass


def test_llama_index_chunk_file_name_eng():
	pass


def test_llama_index_chunk_file_name_eng_node():
	pass
