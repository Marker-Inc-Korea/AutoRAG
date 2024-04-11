import pytest

from autorag.nodes.retrieval.hybrid_cc import cc_pure, hybrid_cc
from tests.autorag.nodes.retrieval.test_hybrid_base import (sample_ids_2, sample_scores_2, sample_ids_3,
                                                            sample_scores_3, pseudo_project_dir,
                                                            base_hybrid_weights_node_test, sample_ids_non_overlap)


def test_cc_pure():
    result_id, result_scores = cc_pure(sample_ids_2, sample_scores_2,
                                       weights=(0.3, 0.7), top_k=3)
    assert result_scores == pytest.approx([1.0, 0.23792372, 0.175])
    assert result_id == ['id-1', 'id-4', 'id-2']


def test_cc_non_overlap():
    result_id, result_scores = cc_pure(sample_ids_non_overlap, sample_scores_2,
                                       weights=(0.3, 0.7), top_k=3)
    assert result_scores == pytest.approx([0.7, 0.3, 0.23792372])
    assert result_id == ['id-6', 'id-1', 'id-4']


def test_hybrid_cc():
    result_id, result_scores = hybrid_cc.__wrapped__(sample_ids_3, sample_scores_3, top_k=3, weights=(0.3, 0.7))
    assert result_id[0] == ['id-1', 'id-4', 'id-2']
    assert result_id[1] == ['id-2', 'id-5', 'id-3']
    assert result_scores[0] == pytest.approx([1.0, 0.23792372, 0.175])
    assert result_scores[1] == pytest.approx([1.0, 0.23792372, 0.175])


def test_hybrid_cc_node(pseudo_project_dir):
    retrieve_scores = [1.0, 0.23792372, 0.175]
    base_hybrid_weights_node_test(hybrid_cc, pseudo_project_dir, retrieve_scores)
