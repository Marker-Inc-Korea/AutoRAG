import math

import pytest

from autorag.evaluate.metric import retrieval_f1, retrieval_precision, retrieval_recall

retrieval_gt = [
    [['test-1', 'test-2'], ['test-3']],
    [['test-4', 'test-5'], ['test-6', 'test-7'], ['test-8']],
    [['test-9', 'test-10']],
    [['test-11'], ['test-12'], ['test-13']]
]

pred = [
    ['test-1', 'pred-1', 'test-2', 'pred-3'],  # recall: 0.5, precision: 0.5, f1: 0.5
    ['test-6', 'pred-5', 'pred-6', 'pred-7'],  # recall: 1/3, precision: 0.25, f1: 2/7
    ['test-9', 'pred-0', 'pred-8', 'pred-9'],  # recall: 1.0, precision: 0.25, f1: 2/5
    ['test-13', 'test-12', 'pred-10', 'pred-11'],  # recall: 2/3, precision: 0.5, f1: 4/7
]


def test_retrieval_f1():
    solution = [0.5, 2 / 7, 2 / 5, 4 / 7]
    result = retrieval_f1(retrieval_gt=retrieval_gt, pred_ids=pred)
    for gt, res in zip(solution, result):
        assert math.isclose(gt, res, rel_tol=1e-4)


def test_retrieval_recall():
    solution = [0.5, 1 / 3, 1, 2 / 3]
    result = retrieval_recall(retrieval_gt=retrieval_gt, pred_ids=pred)
    for gt, res in zip(solution, result):
        assert gt == pytest.approx(res, rel=1e-4)


def test_retrieval_precision():
    solution = [0.5, 0.25, 0.25, 0.5]
    result = retrieval_precision(retrieval_gt=retrieval_gt, pred_ids=pred)
    for gt, res in zip(solution, result):
        assert gt == pytest.approx(res, rel=1e-4)
