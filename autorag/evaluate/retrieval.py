import functools
import warnings
from typing import List, Callable, Any, Tuple

import pandas as pd

from autorag.evaluate.metric import retrieval_recall, retrieval_precision, retrieval_f1


def evaluate_retrieval(retrieval_gt: List[List[List[str]]], metrics: List[str]):
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
            metric_funcs = {
                retrieval_recall.__name__: retrieval_recall,
                retrieval_precision.__name__: retrieval_precision,
                retrieval_f1.__name__: retrieval_f1,
            }

            metric_scores = {}
            for metric in metrics:
                if metric not in metric_funcs:
                    warnings.warn(f"metric {metric} is not in supported metrics: {metric_funcs.keys()}"
                                  f"{metric} will be ignored.")
                else:
                    metric_func = metric_funcs[metric]
                    metric_scores[metric] = metric_func(retrieval_gt=retrieval_gt, pred_ids=pred_ids)

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
