import math

from raground.evaluate.metric import retrieval_f1, retrieval_precision, retrieval_recall

retrieval_gt = [
    ['test-1', 'test-2'],
    ['test-3'],
    ['test-4', 'test-5', 'test-6', 'test-7'],
    ['test-8', 'test-9'],
]

pred = [
    [retrieval_gt[0][0], 'pred-1', 'pred-2', 'pred-3'],  # recall: 0.5, precision: 0.25, f1: 0.333
    ['pred-4', 'pred-5', 'pred-6', 'pred-7'],  # recall: 0, precision: 0, f1: 0
    [retrieval_gt[2][0], retrieval_gt[2][2], 'pred-8', 'pred-9'],  # recall: 0.5, precision: 0.5, f1: 0.5
    [retrieval_gt[3][0], retrieval_gt[3][1], 'pred-10', 'pred-11'],  # recall: 1, precision: 0.5, f1: 0.667
]


def test_retrieval_f1():
    solution = [0.33333, 0, 0.5, 0.66667]
    result = retrieval_f1(retrieval_gt=retrieval_gt, ids=pred)
    for gt, res in zip(solution, result):
        assert math.isclose(gt, res, rel_tol=1e-4)


def test_retrieval_recall():
    solution = [0.5, 0, 0.5, 1]
    result = retrieval_recall(retrieval_gt=retrieval_gt, ids=pred)
    for gt, res in zip(solution, result):
        assert math.isclose(gt, res, rel_tol=1e-4)


def test_retrieval_precision():
    solution = [0.25, 0, 0.5, 0.5]
    result = retrieval_precision(retrieval_gt=retrieval_gt, ids=pred)
    for gt, res in zip(solution, result):
        assert math.isclose(gt, res, rel_tol=1e-4)
