import pytest

from autorag.nodes.retrieval.hybrid_cc import hybrid_cc, fuse_per_query
from tests.autorag.nodes.retrieval.test_hybrid_base import (sample_ids_2, sample_scores_2, sample_ids_3,
                                                            sample_scores_3, sample_retrieval_gt_3, pseudo_project_dir,
                                                            base_hybrid_weights_node_test, sample_ids_non_overlap)


def test_cc_fuse_per_query():
    result_id, result_scores = fuse_per_query(sample_ids_2[0], sample_ids_2[1],
                                              sample_scores_2[0], sample_scores_2[1],
                                              weight=0.3, top_k=3,
                                              normalize_method='mm',
                                              semantic_theoretical_min_value=-1.0,
                                              lexical_theoretical_min_value=0.0)
    assert result_scores == pytest.approx([1.0, 0.23792, 0.175], rel=1e-3)
    assert result_id == ['id-1', 'id-4', 'id-2']


def test_cc_non_overlap():
    result_id, result_scores = fuse_per_query(sample_ids_non_overlap[0], sample_ids_non_overlap[1],
                                              sample_scores_2[0], sample_scores_2[1],
                                              weight=0.3, top_k=3,
                                              normalize_method='mm',
                                              semantic_theoretical_min_value=-1.0,
                                              lexical_theoretical_min_value=0.0
                                              )
    assert result_id == ['id-6', 'id-1', 'id-4']
    assert result_scores == pytest.approx([0.7, 0.3, 0.2379237], rel=1e-3)


def test_hybrid_cc():
    result_id, result_scores, best_weight = hybrid_cc.__wrapped__(sample_ids_3, sample_scores_3, top_k=3,
                                                                  metrics=['retrieval_recall', 'retrieval_ndcg'],
                                                                  retrieval_gt=sample_retrieval_gt_3,
                                                                  normalize_method='tmm')
    assert result_id[0] == ['id-1', 'id-4', 'id-3']
    assert result_id[1] == ['id-2', 'id-5', 'id-4']
    assert result_scores[0] == pytest.approx([1.0, 0.33333, 0.166666], rel=1e-3)
    assert result_scores[1] == pytest.approx([1.0, 0.4285714, 0.2857142], rel=1e-3)
    assert isinstance(best_weight, float)
    assert best_weight == 0.0


def test_hybrid_cc_node(pseudo_project_dir):
    retrieve_scores = [1.0, 0.23792372, 0.175]
    base_hybrid_weights_node_test(hybrid_cc, pseudo_project_dir, retrieve_scores)
