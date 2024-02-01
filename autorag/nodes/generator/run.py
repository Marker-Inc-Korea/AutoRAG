import os
import pathlib
from typing import Callable, List, Dict

import pandas as pd

from autorag.evaluate import evaluate_generation
from autorag.strategy import measure_speed, filter_by_threshold, select_best_average


def run_generator_node(modules: List[Callable],
                       module_params: List[Dict],
                       previous_result: pd.DataFrame,
                       node_line_dir: str,
                       strategies: Dict,
                       ) -> pd.DataFrame:
    """
    Run evaluation and select the best module among generator node results.
    And save the results and summary to generator node directory.

    :param modules: Generator modules to run.
    :param module_params: Generator module parameters.
        Including node parameters, which is used for every module in this node.
    :param previous_result: Previous result dataframe.
        Could be prompt maker node's result.
    :param node_line_dir: This node line's directory.
    :param strategies: Strategies for generator node.
    :return: The best result dataframe.
        It contains previous result columns and generator node's result columns.
    """
    if not os.path.exists(node_line_dir):
        os.makedirs(node_line_dir)
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    node_dir = os.path.join(node_line_dir, "generator")  # node name
    if not os.path.exists(node_dir):
        os.makedirs(node_dir)
    qa_data = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))
    if 'generation_gt' not in qa_data.columns:
        raise ValueError("You must have 'generation_gt' column in qa.parquet.")
    generation_gt = list(map(lambda x: x.tolist(), qa_data['generation_gt'].tolist()))

    results, execution_times = zip(*map(lambda x: measure_speed(
        x[0], project_dir=project_dir, previous_result=previous_result, **x[1]),
                                        zip(modules, module_params)))
    average_times = list(map(lambda x: x / len(results[0]), execution_times))

    if strategies.get('metrics') is None:
        raise ValueError("You must at least one metrics for generator evaluation.")
    results = list(map(lambda result: evaluate_generator_node(result, generation_gt, strategies.get('metrics')), results))

    # save results to folder
    filepaths = list(map(lambda x: os.path.join(node_dir, f'{x}.parquet'), range(len(modules))))
    list(map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths)))  # execute save to parquet
    filenames = list(map(lambda x: os.path.basename(x), filepaths))

    summary_df = pd.DataFrame({
        'filename': filenames,
        'module_name': list(map(lambda module: module.__name__, modules)),
        'module_params': module_params,
        'execution_time': average_times,
        **{metric: list(map(lambda x: x[metric].mean(), results)) for metric in strategies.get('metrics')}
    })

    # filter by strategies
    if strategies.get('speed_threshold') is not None:
        results, filenames = filter_by_threshold(results, average_times, strategies['speed_threshold'], filenames)
    selected_result, selected_filename = select_best_average(results, strategies.get('metrics'), filenames)
    best_result = pd.concat([previous_result, selected_result], axis=1)

    # add 'is_best' column at summary file
    summary_df['is_best'] = summary_df['filename'] == selected_filename

    # save files
    summary_df.to_csv(os.path.join(node_dir, "summary.csv"), index=False)
    best_result.to_parquet(os.path.join(node_dir, f"best_{os.path.splitext(selected_filename)[0]}.parquet"),
                           index=False)
    return best_result


def evaluate_generator_node(result_df: pd.DataFrame, generation_gt, metrics):
    @evaluate_generation(generation_gt=generation_gt, metrics=metrics)
    def evaluate_generation_module(df: pd.DataFrame):
        return df['generated_texts'].tolist(), df['generated_tokens'].tolist(), df['generated_log_probs'].tolist()

    return evaluate_generation_module(result_df)
