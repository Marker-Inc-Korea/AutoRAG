from autorag.nodes.retrieval import hybrid_rsf

from tests.autorag.nodes.retrieval.test_hybrid_base import (sample_ids, sample_scores, base_hybrid_weights_node_test,
                                                            pseudo_project_dir)


def test_hybrid_rsf():
    result_id, result_scores = hybrid_rsf.__wrapped__(sample_ids, sample_scores, top_k=3)
    assert result_id == [
        ['id-3', 'id-1', 'id-2'],
        ['id-4', 'id-2', 'id-3']
    ]
    assert result_scores == [[1.0, 0.25, 0.25], [1.0, 0.25, 0.25]]


def test_hybrid_rsf_dist_based():
    result_id, result_scores = hybrid_rsf.__wrapped__(sample_ids, sample_scores, top_k=3, dist_based=True)
    assert result_id == [
        ['id-3', 'id-1', 'id-2'],
        ['id-4', 'id-2', 'id-3']
    ]
    # Expected scores should be adjusted based on the logic of hybrid_rsf
    assert result_scores == [[0.7041241452319316, 0.3979379273840342, 0.25],
                             [0.7041241452319316, 0.3979379273840342, 0.25]]


def test_hybrid_rsf_node(pseudo_project_dir):
    base_hybrid_weights_node_test(hybrid_rsf, pseudo_project_dir)
