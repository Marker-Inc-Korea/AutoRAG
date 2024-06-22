import functools
import warnings
from typing import List, Callable, Any, Tuple, Optional, Union, Dict

import pandas as pd

from autorag.evaluation.metric import (retrieval_recall, retrieval_precision, retrieval_f1, retrieval_ndcg,
                                       retrieval_mrr,
                                       retrieval_map)
from autorag.evaluation.util import cast_metrics

RETRIEVAL_METRIC_FUNC_DICT = {func.__name__: func for func in
                              [retrieval_recall, retrieval_precision, retrieval_f1, retrieval_ndcg, retrieval_mrr,
                               retrieval_map]}
RETRIEVAL_NO_GT_METRIC_FUNC_DICT = {func.__name__: func for func in []}


def evaluate_retrieval(retrieval_gt: List[List[List[str]]], metrics: Union[List[str], List[Dict]],
                       queries: Optional[List[str]] = None, generation_gt: Optional[List[List[str]]] = None):
    def decorator_evaluate_retrieval(
            func: Callable[[Any], Tuple[List[List[str]], List[List[str]], List[List[float]]]]):
        """
        Decorator for evaluating retrieval results.
        You can use this decorator to any method that returns (contents, scores, ids),
        which is the output of conventional retrieval modules.

        :param func: Must return (contents, scores, ids)
        :return: wrapper function that returns pd.DataFrame, which is the evaluation result.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> pd.DataFrame:
            contents, pred_ids, scores = func(*args, **kwargs)

            metric_scores = {}
            metric_names, metric_params = cast_metrics(metrics)

            for metric_name, metric_param in zip(metric_names, metric_params):
                if metric_name in RETRIEVAL_METRIC_FUNC_DICT:
                    metric_func = RETRIEVAL_METRIC_FUNC_DICT[metric_name]
                    metric_scores[metric_name] = metric_func(retrieval_gt=retrieval_gt, pred_ids=pred_ids,
                                                             **metric_param)
                elif metric_name in RETRIEVAL_NO_GT_METRIC_FUNC_DICT:
                    metric_func = RETRIEVAL_NO_GT_METRIC_FUNC_DICT[metric_name]
                    if queries is None:
                        raise ValueError(f"To using {metric_name}, you have to provide queries.")
                    if generation_gt is None:
                        raise ValueError(f"To using {metric_name}, you have to provide generation ground truth.")
                    metric_scores[metric_name] = metric_func(queries=queries, retrieved_contents=contents,
                                                             generation_gt=generation_gt, **metric_param)
                else:
                    warnings.warn(
                        f"metric {metric_name} is not in supported metrics: {RETRIEVAL_METRIC_FUNC_DICT.keys()}"
                        f" and {RETRIEVAL_NO_GT_METRIC_FUNC_DICT.keys()}"
                        f"{metric_name} will be ignored.")

            metric_result_df = pd.DataFrame(metric_scores)
            execution_result_df = pd.DataFrame({
                'retrieved_contents': contents,
                'retrieved_ids': pred_ids,
                'retrieve_scores': scores,
            })
            result_df = pd.concat([execution_result_df, metric_result_df], axis=1)
            return result_df

        return wrapper

    return decorator_evaluate_retrieval
