import math
from typing import Tuple, List

import pandas as pd

from autorag.evaluate import evaluate_retrieval

retrieval_gt = [[[f'test{i}-{j}'] for i in range(2)] for j in range(4)]


@evaluate_retrieval(retrieval_gt=retrieval_gt, metrics=['retrieval_recall', 'retrieval_precision', 'retrieval_f1'])
def pseudo_retrieval() -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
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
        [retrieval_gt[0][0][0], retrieval_gt[0][1][0], 'pred-0', 'pred-1'],
        ['pred-2', retrieval_gt[1][0][0], retrieval_gt[1][1][0], 'pred-3'],
        [f'pred-{i}' for i in range(4, 8)],
        [retrieval_gt[3][0][0], 'pred-8', 'pred-9', 'pred-10'],
    ]
    return contents, ids, scores


def test_evaluate_retrieval():
    result_df = pseudo_retrieval()
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 4
    assert len(result_df.columns) == 6
    assert list(result_df.columns) == ['retrieved_contents', 'retrieved_ids', 'retrieve_scores', 'retrieval_recall',
                                       'retrieval_precision', 'retrieval_f1']
    recall = result_df['retrieval_recall'].tolist()
    recall_solution = [1, 1, 0, 0.5]
    for pred, sol in zip(recall, recall_solution):
        assert math.isclose(pred, sol, rel_tol=1e-4)

    precision = result_df['retrieval_precision'].tolist()
    precision_solution = [0.5, 0.5, 0, 0.25]
    for pred, sol in zip(precision, precision_solution):
        assert math.isclose(pred, sol, rel_tol=1e-4)

    f1 = result_df['retrieval_f1'].tolist()
    f1_solution = [1 / 1.5, 1 / 1.5, 0, 0.25 / 0.75]
    for pred, sol in zip(f1, f1_solution):
        assert math.isclose(pred, sol, rel_tol=1e-4)
