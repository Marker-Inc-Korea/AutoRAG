import pytest

from autorag.nodes.retrieval.hybrid_cc import cc_pure, hybrid_cc


def test_cc_pure():
    sample_ids = (['id-1', 'id-2', 'id-3', 'id-4', 'id-5'],
                  ['id-1', 'id-4', 'id-3', 'id-5', 'id-2'])
    sample_scores = ([5, 3, 1, 0.4, 0.2], [6, 2, 1, 0.5, 0.1])
    result_id, result_scores = cc_pure(sample_ids, sample_scores,
                                       weights=(0.3, 0.7), top_k=3)
    assert result_scores == pytest.approx([1.0, 0.23792372, 0.175])
    assert result_id == ['id-1', 'id-4', 'id-2']


def test_hybrid_cc():
    sample_ids = ([
                      ['id-1', 'id-2', 'id-3', 'id-4', 'id-5'],
                      ['id-2', 'id-3', 'id-4', 'id-5', 'id-6']
                  ], [
                      ['id-1', 'id-4', 'id-3', 'id-5', 'id-2'],
                      ['id-2', 'id-5', 'id-4', 'id-6', 'id-3']
                  ])
    sample_scores = ([
                         [5, 3, 1, 0.4, 0.2],
                         [6, 4, 2, 1.4, 1.2]
                     ], [
                         [6, 2, 1, 0.5, 0.1],
                         [7, 3, 2, 1.5, 1.1],
                     ])
    result_id, result_scores = hybrid_cc(sample_ids, sample_scores, top_k=3, weights=(0.3, 0.7))
    assert result_id[0] == ['id-1', 'id-4', 'id-2']
    assert result_id[1] == ['id-2', 'id-5', 'id-3']
    assert result_scores[0] == pytest.approx([1.0, 0.23792372, 0.175])
    assert result_scores[1] == pytest.approx([1.0, 0.23792372, 0.175])
