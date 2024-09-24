import pytest

from autorag.nodes.passagereranker import MonoT5
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	base_reranker_test,
	base_reranker_node_test,
	queries_example,
	contents_example,
	ids_example,
	project_dir,
	previous_result,
)
from tests.delete_tests import is_github_action


@pytest.fixture
def monot5_instance():
	return MonoT5(project_dir, "castorini/monot5-3b-msmarco-10k")


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_monot5(monot5_instance):
	top_k = 1
	contents_result, id_result, score_result = monot5_instance._pure(
		queries_example, contents_example, ids_example, top_k
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_monot5_batch_one(monot5_instance):
	top_k = 1
	batch = 1
	contents_result, id_result, score_result = monot5_instance._pure(
		queries_example,
		contents_example,
		ids_example,
		top_k,
		batch=batch,
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_monot5_node():
	top_k = 1
	result_df = MonoT5.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, top_k=top_k
	)
	base_reranker_node_test(result_df, top_k)
