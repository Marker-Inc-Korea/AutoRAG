import itertools

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from autorag.nodes.passagereranker import ColbertReranker
from autorag.nodes.passagereranker.colbert import (
	get_colbert_embedding_batch,
	slice_tensor,
)
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	queries_example,
	contents_example,
	ids_example,
	base_reranker_test,
	project_dir,
	previous_result,
	base_reranker_node_test,
)
from tests.delete_tests import is_github_action


@pytest.fixture
def colbert_reranker_instance():
	return ColbertReranker(project_dir, "colbert-ir/colbertv2.0")


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it uses local model.",
)
def test_colbert_reranker(colbert_reranker_instance):
	top_k = 2
	contents_result, id_result, score_result = colbert_reranker_instance._pure(
		queries_example, contents_example, ids_example, top_k
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it uses local model.",
)
def test_colbert_reranker_long(colbert_reranker_instance):
	top_k = 2
	contents_example[0][0] = contents_example[0][0] * 10000
	contents_result, id_result, score_result = colbert_reranker_instance._pure(
		queries_example, contents_example, ids_example, top_k, batch=2
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it uses local model.",
)
def test_colbert_reranker_one_batch(colbert_reranker_instance):
	top_k = 2
	contents_result, id_result, score_result = colbert_reranker_instance._pure(
		queries_example, contents_example, ids_example, top_k, batch=1
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it uses local model.",
)
def test_colbert_reranker_node():
	top_k = 1
	result_df = ColbertReranker.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, top_k=top_k
	)
	base_reranker_node_test(result_df, top_k)


@pytest.mark.skipif(
	is_github_action(),
	reason="Skipping this test on GitHub Actions because it uses local model.",
)
def test_colbert_embedding():
	contents = list(itertools.chain.from_iterable(contents_example))
	model_name = "colbert-ir/colbertv2.0"
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = AutoModel.from_pretrained(model_name).to(device)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	colbert_embedding = get_colbert_embedding_batch(
		contents, model, tokenizer, batch_size=2
	)

	assert isinstance(colbert_embedding, list)
	assert len(colbert_embedding) == len(contents)
	assert colbert_embedding[0].shape == (1, 11, 768)


def test_slice_tensor():
	original_tensor = torch.randn(14, 7)
	batch_size = 4
	resulting_tensor_list = slice_tensor(original_tensor, batch_size)
	assert len(resulting_tensor_list) == 4
	assert resulting_tensor_list[0].size() == torch.Size([4, 7])
	assert resulting_tensor_list[-1].size() == torch.Size([2, 7])
