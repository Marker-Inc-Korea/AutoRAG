import pytest

from autorag.nodes.passagereranker import FlagEmbeddingLLMReranker
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
def flag_embedding_llm_reranker():
	return FlagEmbeddingLLMReranker(
		project_dir, "BAAI/bge-reranker-v2-gemma", use_fp16=False
	)


@pytest.mark.skip(reason="This test is too slow on the local macbook pro")
@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_flag_embedding_llm_reranker(flag_embedding_llm_reranker):
	top_k = 3
	contents_result, id_result, score_result = flag_embedding_llm_reranker._pure(
		queries_example,
		contents_example,
		ids_example,
		top_k,
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skip(reason="This test is too slow on the local macbook pro")
@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_flag_embedding_llm_reranker_batch_one(flag_embedding_llm_reranker):
	top_k = 3
	batch = 1
	contents_result, id_result, score_result = flag_embedding_llm_reranker._pure(
		queries_example,
		contents_example,
		ids_example,
		top_k,
		batch,
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skip(reason="This test is too slow on the local macbook pro")
@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_flag_embedding_llm_reranker_batch(flag_embedding_llm_reranker):
	top_k = 3
	batch = 2
	contents_result, id_result, score_result = flag_embedding_llm_reranker._pure(
		queries_example,
		contents_example,
		ids_example,
		top_k,
		batch,
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skip(reason="This test is too slow on the local macbook pro")
@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_flag_embedding_llm_reranker_node():
	top_k = 1
	result_df = FlagEmbeddingLLMReranker.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, top_k=top_k
	)
	base_reranker_node_test(result_df, top_k)
