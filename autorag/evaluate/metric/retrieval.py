import functools
import itertools
import math
from typing import List


def retrieval_metric(func):
    @functools.wraps(func)
    def wrapper(retrieval_gt: List[List[List[str]]], pred_ids: List[List[str]]) -> List[float]:
        results = []
        for gt, pred in zip(retrieval_gt, pred_ids):
            if gt == [[]] or any(bool(g_) is False for g in gt for g_ in g):
                results.append(None)
            else:
                results.append(func(gt, pred))
        return results

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


@retrieval_metric
def retrieval_ndcg(gt: List[List[str]], pred: List[str]):
    gt_sets = [frozenset(g) for g in gt]
    pred_set = set(pred)
    relevance_scores = {pred_id: 1 if any(pred_id in gt_set for gt_set in gt_sets) else 0 for pred_id in pred_set}

    dcg = sum((2 ** relevance_scores[doc_id] - 1) / math.log2(i + 2) for i, doc_id in enumerate(pred))

    len_flatten_gt = len(list(itertools.chain.from_iterable(gt)))
    len_pred = len(pred)
    ideal_pred = [1] * min(len_flatten_gt, len_pred) + [0] * max(0, len_pred - len_flatten_gt)
    idcg = sum(relevance / math.log2(i + 2) for i, relevance in enumerate(ideal_pred))

    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg


@retrieval_metric
def retrieval_mrr(gt: List[List[str]], pred: List[str]) -> float:
    """
    Reciprocal Rank (RR) is the reciprocal of the rank of the first relevant item.
    Mean of RR in whole queries is MRR.
    """
    # Flatten the ground truth list of lists into a single set of relevant documents
    gt_sets = [frozenset(g) for g in gt]

    rr_list = []
    for gt_set in gt_sets:
        for i, pred_id in enumerate(pred):
            if pred_id in gt_set:
                rr_list.append(1.0 / (i + 1))
                break
    return sum(rr_list) / len(gt_sets) if rr_list else 0.0


@retrieval_metric
def retrieval_map(gt: List[List[str]], pred: List[str]) -> float:
    """
    Mean Average Precision (MAP) is the mean of Average Precision (AP) for all queries.
    """
    gt_sets = [frozenset(g) for g in gt]

    ap_list = []

    for gt_set in gt_sets:
        pred_hits = [1 if pred_id in gt_set else 0 for pred_id in pred]
        precision_list = [sum(pred_hits[:i + 1]) / (i + 1) for i, hit in enumerate(pred_hits) if hit == 1]
        ap_list.append(sum(precision_list) / len(precision_list) if precision_list else 0.0)

    return sum(ap_list) / len(gt_sets) if ap_list else 0.0
