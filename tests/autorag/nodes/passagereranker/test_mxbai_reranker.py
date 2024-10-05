import pytest

from autorag.nodes.passagereranker import MxBaiReranker
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import (
	queries_example,
	contents_example,
	scores_example,
	ids_example,
	base_reranker_test,
	project_dir,
	previous_result,
	base_reranker_node_test,
)

from tests.delete_tests import is_github_action


@pytest.fixture
def mxbai_reranker_instance():
	return MxBaiReranker(
		project_dir=project_dir, model_name="mixedbread-ai/mxbai-rerank-large-v1"
	)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_mxbai_reranker(mxbai_reranker_instance):
	top_k = 1
	contents_result, id_result, score_result = mxbai_reranker_instance._pure(
		queries_example, contents_example, scores_example, ids_example, top_k
	)
	base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_monot5_node():
	top_k = 1
	result_df = MxBaiReranker.run_evaluator(
		project_dir=project_dir, previous_result=previous_result, top_k=top_k
	)
	base_reranker_node_test(result_df, top_k)
