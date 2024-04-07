import pytest

from autorag.nodes.passagereranker import colbert_reranker
from tests.autorag.nodes.passagereranker.test_passage_reranker_base import queries_example, contents_example, \
    scores_example, ids_example, base_reranker_test, project_dir, previous_result, base_reranker_node_test
from tests.delete_tests import is_github_action


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions because it uses local model.")
def test_colbert_reranker():
    top_k = 2
    original_colbert_reranker = colbert_reranker.__wrapped__
    contents_result, id_result, score_result \
        = original_colbert_reranker(queries_example, contents_example, scores_example, ids_example, top_k)
    base_reranker_test(contents_result, id_result, score_result, top_k)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions because it uses local model.")
def test_colbert_reranker_node():
    top_k = 1
    result_df = colbert_reranker(project_dir=project_dir, previous_result=previous_result, top_k=top_k)
    base_reranker_node_test(result_df, top_k)
