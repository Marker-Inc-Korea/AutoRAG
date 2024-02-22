import pytest

from autorag.nodes.passagereranker import koreranker
from tests.autorag.nodes.passagereranker.test_passage_reranker_base \
    import (base_reranker_test, base_reranker_node_test, ko_queries_example,
            ko_contents_example, scores_example, ids_example, project_dir, ko_previous_result)
from tests.delete_tests import is_github_action


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_koreranker():
    top_k = 1
    original_koreranker = koreranker.__wrapped__
    contents_result, id_result, score_result \
        = original_koreranker(ko_queries_example, ko_contents_example, scores_example, ids_example, top_k)
    base_reranker_test(contents_result, id_result, score_result, top_k, use_ko=True)


@pytest.mark.skipif(is_github_action(), reason="Skipping this test on GitHub Actions")
def test_koreranker_node():
    top_k = 1
    result_df = koreranker(project_dir=project_dir, previous_result=ko_previous_result, top_k=top_k)
    base_reranker_node_test(result_df, top_k, use_ko=True)
