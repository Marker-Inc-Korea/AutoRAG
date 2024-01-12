import functools
import warnings
from typing import List
from uuid import UUID

import pandas as pd

from raground.evaluate.metric import retrieval_recall, retrieval_precision, retrieval_f1


def evaluate_retrieval(func):
    """
    Decorator for evaluating retrieval results.
    You can use this decorator to any method that returns (contents, scores, retrieval_gt),
    which is the output of conventional retrieval modules.

    :param func: Must return (contents, scores, retrieval_gt)
    :return: wrapper function that returns pd.DataFrame, which is the evaluation result.
    """

    @functools.wraps(func)
    def wrapper(retrieval_gt: List[List[UUID]], strategies: List[str], *args, **kwargs) -> pd.DataFrame:
        contents, scores, pred_ids = func(*args, **kwargs)
        strategy_funcs = {
            'recall': retrieval_recall,
            'precision': retrieval_precision,
            'f1': retrieval_f1,
        }

        metric_scores = {}
        for strategy in strategies:
            if strategy not in strategy_funcs:
                warnings.warn(f"strategy {strategy} is not in supported strategies: {strategy_funcs.keys()}"
                              f"{strategy} will be ignored.")
            metric_func = strategy_funcs[strategy]
            metric_scores[strategy] = metric_func(retrieval_gt=retrieval_gt, ids=pred_ids)

        metric_result_df = pd.DataFrame(metric_scores)
        execution_result_df = pd.DataFrame({
            'contents': contents,
            'scores': scores,
            'pred_ids': pred_ids,
            'retrieval_gt': retrieval_gt,
        })
        result_df = pd.concat([execution_result_df, metric_result_df], axis=1)
        return result_df

    return wrapper
