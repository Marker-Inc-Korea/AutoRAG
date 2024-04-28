import functools
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
    relevance_scores = {doc_id: 1 if doc_id in gt[0] else 0 for doc_id in pred}

    dcg = sum((2 ** relevance_scores[doc_id] - 1) / math.log2(i + 2) for i, doc_id in enumerate(pred))
    ideal_sorted_docs = sorted(relevance_scores.items(), key=lambda item: item[1], reverse=True)
    idcg = sum((2 ** relevance - 1) / math.log2(i + 2) for i, (_, relevance) in enumerate(ideal_sorted_docs))
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg
