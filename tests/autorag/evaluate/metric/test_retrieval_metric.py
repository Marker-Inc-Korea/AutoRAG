import pytest

from autorag.evaluate.metric import (retrieval_f1, retrieval_precision, retrieval_recall, retrieval_ndcg, retrieval_mrr,
                                     retrieval_map)

retrieval_gt = [
    [['test-1', 'test-2'], ['test-3']],
    [['test-4', 'test-5'], ['test-6', 'test-7'], ['test-8']],
    [['test-9', 'test-10']],
    [['test-11'], ['test-12'], ['test-13']],
    [['test-14']],
    [[]],
    [['']],
    [['test-15']]
]

pred = [
    ['test-1', 'pred-1', 'test-2', 'pred-3'],  # recall: 0.5, precision: 0.5, f1: 0.5
    ['test-6', 'pred-5', 'pred-6', 'pred-7'],  # recall: 1/3, precision: 0.25, f1: 2/7
    ['test-9', 'pred-0', 'pred-8', 'pred-9'],  # recall: 1.0, precision: 0.25, f1: 2/5
    ['test-13', 'test-12', 'pred-10', 'pred-11'],  # recall: 2/3, precision: 0.5, f1: 4/7
    ['test-14', 'pred-12'],  # recall: 1.0, precision: 0.5, f1: 2/3
    ['pred-13'],  # retrieval_gt is empty so not counted
    ['pred-14'],  # retrieval_gt is empty so not counted
    ['pred-15', 'pred-16', 'test-15']  # recall:1, precision: 1/3, f1: 0.5
]


def test_retrieval_f1():
    solution = [0.5, 2 / 7, 2 / 5, 4 / 7, 2 / 3, None, None, 0.5]
    result = retrieval_f1(retrieval_gt=retrieval_gt, pred_ids=pred)
    for gt, res in zip(solution, result):
        assert gt == pytest.approx(res, rel=1e-4)


def test_retrieval_recall():
    solution = [0.5, 1 / 3, 1, 2 / 3, 1, None, None, 1]
    result = retrieval_recall(retrieval_gt=retrieval_gt, pred_ids=pred)
    for gt, res in zip(solution, result):
        assert gt == pytest.approx(res, rel=1e-4)


def test_retrieval_precision():
    solution = [0.5, 0.25, 0.25, 0.5, 0.5, None, None, 1 / 3]
    result = retrieval_precision(retrieval_gt=retrieval_gt, pred_ids=pred)
    for gt, res in zip(solution, result):
        assert gt == pytest.approx(res, rel=1e-4)


def test_retrieval_ndcg():
    solution = [0.7039180890341347, 0.3903800499921017, 0.6131471927654584, 0.7653606369886217, 1, None, None, 0.5]
    result = retrieval_ndcg(retrieval_gt=retrieval_gt, pred_ids=pred)
    for gt, res in zip(solution, result):
        assert gt == pytest.approx(res, rel=1e-4)


def test_retrieval_mrr():
    solution = [1 / 2, 1 / 3, 1, 1 / 2, 1, None, None, 1 / 3]
    result = retrieval_mrr(retrieval_gt=retrieval_gt, pred_ids=pred)
    for gt, res in zip(solution, result):
        assert gt == pytest.approx(res, rel=1e-4)


def test_retrieval_map():
    solution = [5 / 12, 1 / 3, 1, 1 / 2, 1, None, None, 1 / 3]
    result = retrieval_map(retrieval_gt=retrieval_gt, pred_ids=pred)
    for gt, res in zip(solution, result):
        assert gt == pytest.approx(res, rel=1e-4)
