import pytest

from autorag.nodes.retrieval.hybrid_rrf import rrf_pure
from autorag.nodes.retrieval import hybrid_rrf


def test_hybrid_rrf():
    sample_ids = ([
                      ['id-1', 'id-2', 'id-3'],
                      ['id-2', 'id-3', 'id-4']
                  ], [
                      ['id-1', 'id-4', 'id-3'],
                      ['id-2', 'id-5', 'id-4']
                  ])
    sample_scores = ([
                         [1, 3, 5],
                         [2, 4, 6]
                     ], [
                         [4, 2, 6],
                         [5, 3, 7]
                     ])
    result_id, result_scores = hybrid_rrf(sample_ids, sample_scores, top_k=3, rrf_k=1)
    assert result_id == [
        ['id-3', 'id-1', 'id-2'],
        ['id-4', 'id-2', 'id-3']
    ]
    assert result_scores[0] == pytest.approx([1.0, (1 / 4) + (1 / 3), (1 / 3)])
    assert result_scores[1] == pytest.approx([1.0, (1 / 4) + (1 / 3), (1 / 3)])


def test_rrf_pure():
    sample_ids = (['id-1', 'id-2', 'id-3'],
                  ['id-1', 'id-4', 'id-3'])
    sample_scores = ([1, 3, 5], [4, 2, 6])
    result_id, result_scores = rrf_pure(sample_ids, sample_scores,
                                        rrf_k=1, top_k=3)
    assert result_scores == pytest.approx([1.0, (1 / 4) + (1 / 3), (1 / 3)])
    assert result_id == ['id-3', 'id-1', 'id-2']
