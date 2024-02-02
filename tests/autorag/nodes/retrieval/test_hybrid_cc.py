import pytest

from autorag.nodes.retrieval.hybrid_cc import cc_pure


def test_cc_pure():
    sample_ids = (['id-1', 'id-2', 'id-3', 'id-4', 'id-5'],
                  ['id-1', 'id-4', 'id-3', 'id-5', 'id-2'])
    sample_scores = ([5, 3, 1, 0.4, 0.2], [6, 2, 1, 0.5, 0.1])
    result_id, result_scores = cc_pure(sample_ids, sample_scores,
                                       weights=(0.3, 0.7), top_k=3)
    assert result_scores == pytest.approx([1.0, 0.23792372, 0.175])
    assert result_id == ['id-1', 'id-4', 'id-2']
