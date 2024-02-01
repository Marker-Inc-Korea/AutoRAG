"""
This file contains the retrieval contents metric,
which means calculate the metric based on the contents of the retrieved items.
"""
import functools
import itertools
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import swifter

from autorag.utils.util import normalize_string


def retrieval_contents_metric(func):
    @functools.wraps(func)
    def wrapper(gt_contents: List[List[str]], pred_contents: List[List[str]]) -> List[float]:
        df = pd.DataFrame({'gt': gt_contents, 'pred': pred_contents})
        df[func.__name__] = df.swifter.apply(lambda x: func(x['gt'], x['pred']), axis=1)
        return df[func.__name__].tolist()

    return wrapper


def single_token_f1(ground_truth: str, prediction: str):
    prediction_tokens = normalize_string(prediction).split()
    ground_truth_tokens = normalize_string(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


@retrieval_contents_metric
def retrieval_token_f1(gt: List[str], pred: List[str]):
    calculated_results = list(map(lambda x: single_token_f1(x[1], x[0]), list(itertools.product(pred, gt))))
    _, _, result = zip(*calculated_results)
    result_np = np.array(list(result)).reshape(len(pred), -1)
    return result_np.max(axis=1).mean()


@retrieval_contents_metric
def retrieval_token_precision(gt: List[str], pred: List[str]):
    calculated_results = list(map(lambda x: single_token_f1(x[1], x[0]), list(itertools.product(pred, gt))))
    result, _, _ = zip(*calculated_results)
    result_np = np.array(list(result)).reshape(len(pred), -1)
    return result_np.max(axis=1).mean()


@retrieval_contents_metric
def retrieval_token_recall(gt: List[str], pred: List[str]):
    calculated_results = list(map(lambda x: single_token_f1(x[1], x[0]), list(itertools.product(pred, gt))))
    _, result, _ = zip(*calculated_results)
    result_np = np.array(list(result)).reshape(len(pred), -1)
    return result_np.max(axis=1).mean()
