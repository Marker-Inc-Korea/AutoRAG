import pytest

from autorag.nodes.passagereranker import Upr
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	base_reranker_test,
	queries_example,
	contents_example,
	ids_example,
	base_reranker_node_test,
	project_dir,
	previous_result,
)
from tests.delete_tests import is_github_action


@pytest.fixture
def upr_reranker():
	return Upr(project_dir)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_upr(upr_reranker):
	top_k = 1
	contents_result, id_result, score_result = upr_reranker._pure(
		queries_example, contents_example, ids_example, top_k
	)
	base_reranker_test(
		contents_result, id_result, score_result, top_k, descending=False
	)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_upr_node():
	top_k = 3
	result_df = Upr.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, top_k=top_k
	)
	base_reranker_node_test(result_df, top_k, descending=False)
