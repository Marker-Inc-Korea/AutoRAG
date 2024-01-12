import math
from typing import Tuple, List
from uuid import UUID, uuid4

import pandas as pd

from raground.evaluate import evaluate_retrieval

retrieval_gt = [[uuid4() for _ in range(2)] for _ in range(4)]


@evaluate_retrieval
def pseudo_retrieval() -> Tuple[List[List[str]], List[List[float]], List[List[UUID]]]:
    contents = [
        ['a', 'b', 'c', 'd'],
        ['better', 'bone', 'caleb', 'done'],
        ['ate', 'better', 'cortex', 'dad'],
        ['aim', 'bond', 'cane', 'dance'],
    ]
    scores = [
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
        [0.5, 0.4, 0.3, 0.2],
        [0.6, 0.5, 0.4, 0.3],
    ]
    ids = [
        [retrieval_gt[0][0], retrieval_gt[0][1], uuid4(), uuid4()],
        [uuid4(), retrieval_gt[1][0], retrieval_gt[1][1], uuid4()],
        [uuid4() for _ in range(4)],
        [retrieval_gt[3][0], uuid4(), uuid4(), uuid4()],
    ]
    return contents, scores, ids


def test_evaluate_retrieval():
    strategies = ['recall', 'precision', 'f1']
    result_df = pseudo_retrieval(retrieval_gt=retrieval_gt, strategies=strategies)
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 4
    assert len(result_df.columns) == 7
    assert list(result_df.columns) == ['contents', 'scores', 'pred_ids', 'retrieval_gt', 'recall', 'precision', 'f1']
    recall = result_df['recall'].tolist()
    recall_solution = [1, 1, 0, 0.5]
    for pred, sol in zip(recall, recall_solution):
        assert math.isclose(pred, sol, rel_tol=1e-4)

    precision = result_df['precision'].tolist()
    precision_solution = [0.5, 0.5, 0, 0.25]
    for pred, sol in zip(precision, precision_solution):
        assert math.isclose(pred, sol, rel_tol=1e-4)

    f1 = result_df['f1'].tolist()
    f1_solution = [1 / 1.5, 1 / 1.5, 0, 0.25 / 0.75]
    for pred, sol in zip(f1, f1_solution):
        assert math.isclose(pred, sol, rel_tol=1e-4)
