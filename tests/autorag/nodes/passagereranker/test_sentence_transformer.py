import pytest

from autorag.nodes.passagereranker import SentenceTransformerReranker
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
def sentence_transformer_instance():
	return SentenceTransformerReranker(
		project_dir, "cross-encoder/ms-marco-MiniLM-L-2-v2"
	)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_sentence_transformer_reranker(sentence_transformer_instance):
	top_k = 3
	contents_result, id_result, score_result = sentence_transformer_instance._pure(
		queries_example, contents_example, ids_example, top_k
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_sentence_transformer_reranker_batch_one(sentence_transformer_instance):
	top_k = 3
	batch = 1
	contents_result, id_result, score_result = sentence_transformer_instance._pure(
		queries_example, contents_example, ids_example, top_k, batch
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_sentence_transformer_reranker_node():
	top_k = 1
	result_df = SentenceTransformerReranker.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, top_k=top_k
	)
	base_reranker_node_test(result_df, top_k)
