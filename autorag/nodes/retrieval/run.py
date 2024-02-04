import logging
import os
import pathlib
from typing import List, Callable, Dict, Tuple

import pandas as pd

from autorag.evaluate import evaluate_retrieval
from autorag.strategy import measure_speed, filter_by_threshold, select_best_average
from autorag.utils.util import load_summary_file

logger = logging.getLogger("AutoRAG")


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
    save_dir = os.path.join(node_line_dir, "retrieval")  # node name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def run_and_save(input_modules, input_module_params, filename_start: int):
        result, execution_times = zip(*map(lambda task: measure_speed(
            task[0], project_dir=project_dir, previous_result=previous_result, **task[1]),
                                           zip(input_modules, input_module_params)))
        average_times = list(map(lambda x: x / len(result[0]), execution_times))

        # run metrics before filtering
        if strategies.get('metrics') is None:
            raise ValueError("You must at least one metrics for retrieval evaluation.")
        result = list(map(lambda x: evaluate_retrieval_node(x, retrieval_gt, strategies.get('metrics')), result))

        # save results to folder
        filepaths = list(map(lambda x: os.path.join(save_dir, f'{x}.parquet'),
                             range(filename_start, filename_start + len(input_modules))))
        list(map(lambda x: x[0].to_parquet(x[1], index=False), zip(result, filepaths)))  # execute save to parquet
        filename_list = list(map(lambda x: os.path.basename(x), filepaths))

        summary_df = pd.DataFrame({
            'filename': filename_list,
            'module_name': list(map(lambda module: module.__name__, input_modules)),
            'module_params': input_module_params,
            'execution_time': average_times,
            **{metric: list(map(lambda result: result[metric].mean(), result)) for metric in
               strategies.get('metrics')},
        })
        summary_df.to_csv(os.path.join(save_dir, 'summary.csv'), index=False)
        return result, average_times, summary_df

    # run retrieval modules except hybrid
    hybrid_module_names = ['hybrid_rrf', 'hybrid_cc']
    filename_first = 0
    if any([module.__name__ not in hybrid_module_names for module in modules]):
        non_hybrid_modules, non_hybrid_module_params = zip(*filter(lambda x: x[0].__name__ not in hybrid_module_names,
                                                                   zip(modules, module_params)))
        non_hybrid_results, non_hybrid_times, non_hybrid_summary_df = run_and_save(non_hybrid_modules,
                                                                                   non_hybrid_module_params, filename_first)
        filename_first += len(non_hybrid_modules)
    else:
        non_hybrid_results, non_hybrid_times, non_hybrid_summary_df = [], [], pd.DataFrame()

    if any([module.__name__ in hybrid_module_names for module in modules]):
        hybrid_modules, hybrid_module_params = zip(*filter(lambda x: x[0].__name__ in hybrid_module_names,
                                                           zip(modules, module_params)))
        if all(['target_module_params' in x for x in hybrid_module_params]):
            # If target_module_params are already given, run hybrid retrieval directly
            hybrid_results, hybrid_times, hybrid_summary_df = run_and_save(hybrid_modules, hybrid_module_params,
                                                                           filename_first)
            filename_first += len(hybrid_modules)
        else:
            target_modules = list(map(lambda x: x.pop('target_modules'), hybrid_module_params))
            target_filenames = list(map(lambda x: select_result_for_hybrid(save_dir, x), target_modules))
            ids_scores = list(map(lambda x: get_ids_and_scores(save_dir, x), target_filenames))
            target_module_params = list(map(lambda x: get_module_params(save_dir, x), target_filenames))
            hybrid_module_params = list(map(lambda x: {**x[0], **x[1]}, zip(hybrid_module_params, ids_scores)))
            real_hybrid_times = list(map(lambda filename: get_hybrid_execution_times(save_dir, filename), target_filenames))
            hybrid_results, hybrid_times, hybrid_summary_df = run_and_save(hybrid_modules, hybrid_module_params,
                                                                           filename_first)
            filename_first += len(hybrid_modules)
            hybrid_times = real_hybrid_times.copy()
            hybrid_summary_df['execution_time'] = hybrid_times
            hybrid_summary_df = edit_summary_df_params(hybrid_summary_df, target_modules, target_module_params)
    else:
        hybrid_results, hybrid_times, hybrid_summary_df = [], [], pd.DataFrame()

    summary = pd.concat([non_hybrid_summary_df, hybrid_summary_df], ignore_index=True)
    results = non_hybrid_results + hybrid_results
    average_times = non_hybrid_times + hybrid_times
    filenames = summary['filename'].tolist()

    # filter by strategies
    if strategies.get('speed_threshold') is not None:
        results, filenames = filter_by_threshold(results, average_times, strategies['speed_threshold'], filenames)
    selected_result, selected_filename = select_best_average(results, strategies.get('metrics'), filenames)
    best_result = pd.concat([previous_result, selected_result], axis=1)

    # add summary.csv 'is_best' column
    summary['is_best'] = summary['filename'] == selected_filename

    # save the result files
    best_result.to_parquet(os.path.join(save_dir, f'best_{os.path.splitext(selected_filename)[0]}.parquet'),
                           index=False)
    summary.to_csv(os.path.join(save_dir, 'summary.csv'), index=False)
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


def select_result_for_hybrid(node_dir: str, target_modules: Tuple) -> List[str]:
    """
    Get ids and scores of target_module from summary.csv and each result parquet file.

    :param node_dir: The directory of the node.
    :param target_modules: The name of the target modules.
    :return: A list of filenames.
    """

    def select_best_among_module(df: pd.DataFrame, module_name: str):
        modules_summary = df.loc[lambda row: row['module_name'] == module_name]
        if len(modules_summary) == 1:
            return modules_summary.iloc[0, :]
        elif len(modules_summary) <= 0:
            raise ValueError(f"module_name {module_name} does not exist in summary.csv. "
                             f"You must run {module_name} before running hybrid retrieval.")
        metrics = modules_summary.drop(columns=['filename', 'module_name', 'module_params', 'execution_time'])
        metric_average = metrics.mean(axis=1)
        metric_average = metric_average.reset_index(drop=True)
        max_idx = metric_average.idxmax()
        best_module = modules_summary.iloc[max_idx, :]
        return best_module

    summary_df = load_summary_file(os.path.join(node_dir, "summary.csv"))
    best_results = list(map(lambda module_name: select_best_among_module(summary_df, module_name), target_modules))
    best_filenames = list(map(lambda df: df['filename'], best_results))
    return best_filenames


def get_module_params(node_dir: str, filenames: List[str]) -> Tuple[Dict]:
    summary_df = load_summary_file(os.path.join(node_dir, "summary.csv"))
    best_results = summary_df[summary_df['filename'].isin(filenames)]
    module_params = best_results['module_params'].tolist()
    return tuple(module_params)


def edit_summary_df_params(summary_df: pd.DataFrame, target_modules, target_module_params) -> pd.DataFrame:
    def delete_ids_scores(x):
        del x['ids']
        del x['scores']
        return x

    summary_df['module_params'] = summary_df['module_params'].apply(delete_ids_scores)
    summary_df['new_params'] = [{'target_modules': x, 'target_module_params': y} for x, y in zip(target_modules, target_module_params)]
    summary_df['module_params'] = summary_df.apply(lambda row: {**row['module_params'], **row['new_params']}, axis=1)
    summary_df = summary_df.drop(columns=['new_params'])
    return summary_df


def get_ids_and_scores(node_dir: str, filenames: List[str]) -> Dict:
    best_results_df = list(map(lambda filename: pd.read_parquet(os.path.join(node_dir, filename)), filenames))
    ids = tuple(map(lambda df: df['retrieved_ids'].apply(list).tolist(), best_results_df))
    scores = tuple(map(lambda df: df['retrieve_scores'].apply(list).tolist(), best_results_df))
    return {
        'ids': ids,
        'scores': scores,
    }


def get_hybrid_execution_times(node_dir: str, filenames: List[str]) -> float:
    summary_df = load_summary_file(os.path.join(node_dir, "summary.csv"))
    best_results = summary_df[summary_df['filename'].isin(filenames)]
    execution_times = best_results['execution_time'].sum()
    return execution_times
