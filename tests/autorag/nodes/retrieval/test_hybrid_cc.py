import pandas as pd
import pytest

from autorag.nodes.retrieval.hybrid_cc import cc_pure, hybrid_cc
from tests.autorag.nodes.retrieval.test_run_retrieval_node import pseudo_node_dir
from tests.autorag.nodes.retrieval.test_hybrid_rrf import pseudo_project_dir


def test_cc_pure():
    sample_ids = (['id-1', 'id-2', 'id-3', 'id-4', 'id-5'],
                  ['id-1', 'id-4', 'id-3', 'id-5', 'id-2'])
    sample_scores = ([5, 3, 1, 0.4, 0.2], [6, 2, 1, 0.5, 0.1])
    result_id, result_scores = cc_pure(sample_ids, sample_scores,
                                       weights=(0.3, 0.7), top_k=3)
    assert result_scores == pytest.approx([1.0, 0.23792372, 0.175])
    assert result_id == ['id-1', 'id-4', 'id-2']


def test_cc_non_overlap():
    sample_ids = (['id-1', 'id-2', 'id-3', 'id-4', 'id-5'],
                  ['id-6', 'id-4', 'id-3', 'id-7', 'id-2'])
    sample_scores = ([5, 3, 1, 0.4, 0.2], [6, 2, 1, 0.5, 0.1])
    result_id, result_scores = cc_pure(sample_ids, sample_scores,
                                       weights=(0.3, 0.7), top_k=3)
    assert result_scores == pytest.approx([0.7, 0.3, 0.23792372])
    assert result_id == ['id-6', 'id-1', 'id-4']


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
    result_id, result_scores = hybrid_cc.__wrapped__(sample_ids, sample_scores, top_k=3, weights=(0.3, 0.7))
    assert result_id[0] == ['id-1', 'id-4', 'id-2']
    assert result_id[1] == ['id-2', 'id-5', 'id-3']
    assert result_scores[0] == pytest.approx([1.0, 0.23792372, 0.175])
    assert result_scores[1] == pytest.approx([1.0, 0.23792372, 0.175])


def test_hybrid_cc_node(pseudo_project_dir, pseudo_node_dir):
    previous_result = pd.DataFrame({
        'qid': ['query-1', 'query-2', 'query-3'],
        'query': ['query-1', 'query-2', 'query-3'],
        'retrieval_gt': [
            [['id-1'], ['id-2'], ['id-3']],
            [['id-1'], ['id-2'], ['id-3']],
            [['id-1'], ['id-2'], ['id-3']]
        ],
        'generation_gt': [
            ['gen-1', 'gen-2'],
            ['gen-1', 'gen-2'],
            ['gen-1', 'gen-2']
        ]
    })
    modules = {
        'ids': ([['id-1', 'id-2', 'id-3', 'id-4', 'id-5'],
                 ['id-1', 'id-2', 'id-3', 'id-4', 'id-5']],
                [['id-1', 'id-4', 'id-3', 'id-5', 'id-2'],
                 ['id-1', 'id-4', 'id-3', 'id-5', 'id-2']]
                ),
        'scores': ([[5, 3, 1, 0.4, 0.2], [5, 3, 1, 0.4, 0.2]],
                   [[6, 2, 1, 0.5, 0.1], [6, 2, 1, 0.5, 0.1]]),
        'top_k': 3,
        'weights': (0.3, 0.7)
    }
    result = hybrid_cc(project_dir=pseudo_project_dir, previous_result=previous_result, **modules)
    assert len(result) == 2
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'retrieved_contents', 'retrieved_ids', 'retrieve_scores'}
    assert result['retrieved_ids'].tolist()[0] == ['id-1', 'id-4', 'id-2']
    assert result['retrieve_scores'].tolist()[0] == pytest.approx([1.0, 0.23792372, 0.175])
    assert result['retrieved_contents'].tolist()[0] == ['doc-1', 'doc-4', 'doc-2']
