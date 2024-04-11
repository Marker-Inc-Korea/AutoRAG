import pytest

from autorag.nodes.retrieval import hybrid_dbsf
from tests.autorag.nodes.retrieval.test_hybrid_base import (sample_ids, sample_scores, base_hybrid_weights_node_test,
                                                            pseudo_project_dir)


def test_hybrid_dbsf():
    result_id, result_scores = hybrid_dbsf.__wrapped__(sample_ids, sample_scores, top_k=3)
    assert result_id == [
        ['id-3', 'id-1', 'id-2'],
        ['id-4', 'id-2', 'id-3']
    ]
    assert result_scores[0] == pytest.approx([0.7041241452319316, 0.3979379273840342, 0.25])
    assert result_scores[1] == pytest.approx([0.7041241452319316, 0.3979379273840342, 0.25])


def test_hybrid_dbsf_node(pseudo_project_dir):
    retrieve_scores = [0.8068646915692769, 0.4628671097814567, 0.4301143181548378]
    base_hybrid_weights_node_test(hybrid_dbsf, pseudo_project_dir, retrieve_scores)
