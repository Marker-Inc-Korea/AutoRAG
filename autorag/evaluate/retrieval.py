import functools
import warnings
from typing import List, Callable, Any, Tuple, Optional

import pandas as pd

from autorag.evaluate.metric import (retrieval_recall, retrieval_precision, retrieval_f1, retrieval_ndcg, retrieval_mrr,
                                     retrieval_map)
from autorag.evaluate.metric.retrieval_no_gt import ragas_context_precision


def evaluate_retrieval(retrieval_gt: List[List[List[str]]], metrics: List[str],
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
            metric_funcs = {
                retrieval_recall.__name__: retrieval_recall,
                retrieval_precision.__name__: retrieval_precision,
                retrieval_f1.__name__: retrieval_f1,
                retrieval_ndcg.__name__: retrieval_ndcg,
                retrieval_mrr.__name__: retrieval_mrr,
                retrieval_map.__name__: retrieval_map,
            }

            no_gt_metric_funcs = {
                ragas_context_precision.__name__: ragas_context_precision,
            }

            metric_scores = {}
            for metric in metrics:
                if metric in metric_funcs:
                    metric_func = metric_funcs[metric]
                    metric_scores[metric] = metric_func(retrieval_gt=retrieval_gt, pred_ids=pred_ids)
                elif metric in no_gt_metric_funcs:
                    metric_func = no_gt_metric_funcs[metric]
                    if queries is None:
                        raise ValueError(f"To using {metric}, you have to provide queries.")
                    if generation_gt is None:
                        raise ValueError(f"To using {metric}, you have to provide generation ground truth.")
                    metric_scores[metric] = metric_func(queries=queries, retrieved_contents=contents,
                                                        generation_gt=generation_gt)
                else:
                    warnings.warn(f"metric {metric} is not in supported metrics: {metric_funcs.keys()}"
                                  f" and {no_gt_metric_funcs.keys()}"
                                  f"{metric} will be ignored.")

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
