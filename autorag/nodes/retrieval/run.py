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
                       strategies: Dict,
                       ) -> pd.DataFrame:
    """
    Run evaluation and select the best module among retrieval node results.

    :param modules: Retrieval modules to run.
    :param module_params: Retrieval module parameters.
    :param previous_result: Previous result dataframe.
        Could be query expansion's best result or qa data.
    :param node_line_dir: This node line's directory.
    :param strategies: Strategies for retrieval node.
    :return: The best result dataframe.
        It contains previous result columns and retrieval node's result columns.
    """
    if not os.path.exists(node_line_dir):
        os.makedirs(node_line_dir)
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    retrieval_gt = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))['retrieval_gt'].tolist()

    results, execution_times = zip(*map(lambda task: measure_speed(
        task[0], project_dir=project_dir, previous_result=previous_result, **task[1]), zip(modules, module_params)))
    average_times = list(map(lambda x: x / len(results[0]), execution_times))

    # run metrics before filtering
    if strategies.get('metrics') is None:
        raise ValueError("You must at least one metrics for retrieval evaluation.")
    results = list(map(lambda x: evaluate_retrieval_node(x, retrieval_gt, strategies.get('metrics')), results))

    # save results to folder
    save_dir = os.path.join(node_line_dir, "retrieval")  # node name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepaths = list(map(lambda x: os.path.join(save_dir, make_module_file_name(x[0].__name__, x[1])),
                         zip(modules, module_params)))
    list(map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths)))  # execute save to parquet

    summary_df = pd.DataFrame({
        'filename': list(map(lambda x: os.path.basename(x), filepaths)),
        **{metric: list(map(lambda result: result[metric].mean(), results)) for metric in strategies.get('metrics')},
    })
    summary_df.to_parquet(os.path.join(save_dir, 'summary.parquet'), index=False)

    # filter by strategies
    if strategies.get('speed_threshold') is not None:
        results = filter_by_threshold(results, average_times, strategies['speed_threshold'])
    selected_result = select_best_average(results, strategies.get('metrics'))
    best_result = pd.concat([previous_result, selected_result], axis=1)

    # save the best result to best.parquet
    best_result.to_parquet(os.path.join(save_dir, 'best.parquet'), index=False)
    return best_result


def evaluate_retrieval_node(result_df: pd.DataFrame, retrieval_gt, metrics) -> pd.DataFrame:
    """
    Evaluate retrieval node from retrieval node result dataframe.

    :param result_df: The result dataframe from a retrieval node.
    :param retrieval_gt: Ground truth for retrieval from qa dataset.
    :param metrics: Metric list from input strategies.
    :return: Return result_df with metrics columns.
        The columns will be 'retrieved_contents', 'retrieved_ids', 'retrieve_scores', and metric names.
    """

    @evaluate_retrieval(retrieval_gt=retrieval_gt, metrics=metrics)
    def evaluate_this_module(df: pd.DataFrame):
        return df['retrieved_contents'].tolist(), df['retrieved_ids'].tolist(), df['retrieve_scores'].tolist()

    return evaluate_this_module(result_df)
