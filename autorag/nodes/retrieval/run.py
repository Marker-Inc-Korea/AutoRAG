import os
import pathlib
from typing import List, Callable, Dict

import pandas as pd

from autorag.evaluate import evaluate_retrieval
from autorag.strategy import measure_speed, filter_by_threshold, select_best_average
from autorag.utils.util import make_module_file_name


def run_retrieval_node(modules: List[Callable],
                       module_params: List[Dict],
                       previous_result: pd.DataFrame,
                       node_line_dir: str,
                       retrieval_gt: pd.DataFrame,
                       strategies: Dict,
                       ) -> pd.DataFrame:
    if not os.path.exists(node_line_dir):
        os.makedirs(node_line_dir)
    project_dir = pathlib.PurePath(node_line_dir).parent.parent

    results, execution_times = zip(*map(lambda task: measure_speed(
        task[0], project_dir=project_dir, previous_result=previous_result, **task[1]), zip(modules, module_params)))
    average_times = list(map(lambda x: x / len(results[0]), execution_times))

    # run metrics before filtering
    if strategies.get('metrics') is None:
        raise ValueError("You must at least one metrics for retrieval evaluation.")
    results = list(map(lambda x: evaluate_retrieval_node(x, retrieval_gt, strategies.get('metrics')), results))

    # save results to folder
    save_dir = os.path.join(node_line_dir, "retrieval")  # node name
    filepaths = list(map(lambda x: os.path.join(save_dir, make_module_file_name(x[0].__name__, x[1])),
                         zip(modules, module_params)))
    map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths))

    # make summary and save it to summary.parquet

    # filter by strategies
    if strategies.get('speed_threshold') is not None:
        results = filter_by_threshold(results, average_times, strategies['speed_threshold'])
    final_result = select_best_average(results, strategies.get('metrics'))
    return final_result


def evaluate_retrieval_node(result_df: pd.DataFrame, retrieval_gt, metrics) -> pd.DataFrame:
    @evaluate_retrieval(retrieval_gt=retrieval_gt, metrics=metrics)
    def evaluate_this_module(df: pd.DataFrame):
        return df['retrieved_contents'].tolist(), df['retrieved_ids'].tolist(), df['retrieve_scores'].tolist()

    return evaluate_this_module(result_df)
