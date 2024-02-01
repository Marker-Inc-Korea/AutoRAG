import functools
from typing import List

import pandas as pd
import swifter  # do not delete this line


def retrieval_metric(func):
    @functools.wraps(func)
    def wrapper(retrieval_gt: List[List[List[str]]], pred_ids: List[List[str]]) -> List[float]:
        # make retrieval_gt and ids to pd dataframe
        df = pd.DataFrame({'gt': retrieval_gt, 'pred': pred_ids})
        df[func.__name__] = df.swifter.apply(lambda x: func(x['gt'], x['pred']), axis=1)
        return df[func.__name__].tolist()

    return wrapper


@retrieval_metric
def retrieval_f1(gt: List[List[str]], pred: List[str]):
    """
    Compute f1 score for retrieval.

    :param gt: 2-d list of ground truth ids.
        It contains and/or connections between ids.
    :param pred: Prediction ids.
    :return: The f1 score.
    """
    recall_score = retrieval_recall.__wrapped__(gt, pred)
    precision_score = retrieval_precision.__wrapped__(gt, pred)
    if recall_score + precision_score == 0:
        return 0
    else:
        return 2 * (recall_score * precision_score) / (recall_score + precision_score)


@retrieval_metric
def retrieval_recall(gt: List[List[str]], pred: List[str]):
    gt_sets = [frozenset(g) for g in gt]
    pred_set = set(pred)
    hits = sum(any(pred_id in gt_set for pred_id in pred_set) for gt_set in gt_sets)
    recall = hits / len(gt) if len(gt) > 0 else 0.0
    return recall


@retrieval_metric
def retrieval_precision(gt: List[List[str]], pred: List[str]):
    gt_sets = [frozenset(g) for g in gt]
    pred_set = set(pred)
    hits = sum(any(pred_id in gt_set for gt_set in gt_sets) for pred_id in pred_set)
    precision = hits / len(pred) if len(pred) > 0 else 0.0
    return precision
