import logging
import os
import pathlib
from typing import List, Callable, Dict, Optional
from copy import deepcopy

import pandas as pd

from autorag import embedding_models
from autorag.nodes.retrieval import bm25, vectordb
from autorag.nodes.retrieval.base import load_bm25_corpus, load_chroma_collection
from autorag.nodes.retrieval.run import evaluate_retrieval_node
from autorag.schema import Module
from autorag.strategy import measure_speed, filter_by_threshold, select_best_average
from autorag.utils.util import make_module_file_name, make_combinations, explode
from autorag.support import get_support_modules

logger = logging.getLogger("AutoRAG")


def run_query_expansion_node(modules: List[Callable],
                             module_params: List[Dict],
                             previous_result: pd.DataFrame,
                             node_line_dir: str,
                             strategies: Dict,
                             ) -> pd.DataFrame:
    """
    Run evaluation and select the best module among query expansion node results.
    Initially, retrieval is run using expanded_queries, the result of the query_expansion module.
    The retrieval module is run as a combination of the retrieval_modules in strategies.
    If there are multiple retrieval_modules, run them all and choose the best result.
    If there are no retrieval_modules, run them with the default of bm25.
    In this way, the best result is selected for each module, and then the best result is selected.

    :param modules: Query expansion modules to run.
    :param module_params: Query expansion module parameters.
    :param previous_result: Previous result dataframe.
        In this case, it would be qa data.
    :param node_line_dir: This node line's directory.
    :param strategies: Strategies for query expansion node.
    :return: The best result dataframe.
    """
    if not os.path.exists(node_line_dir):
        os.makedirs(node_line_dir)
    node_dir = os.path.join(node_line_dir, "query_expansion")
    if not os.path.exists(node_dir):
        os.makedirs(node_dir)
    project_dir = pathlib.PurePath(node_line_dir).parent.parent

    # run query expansion
    results, execution_times = zip(*map(lambda task: measure_speed(
        task[0], project_dir=project_dir, previous_result=previous_result, **task[1]), zip(modules, module_params)))
    average_times = list(map(lambda x: x / len(results[0]), execution_times))

    # save results to folder
    filepaths = list(map(lambda x: os.path.join(node_dir, make_module_file_name(x[0].__name__, x[1])),
                         zip(modules, module_params)))
    list(map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths)))  # execute save to parquet
    filenames = list(map(lambda x: os.path.basename(x), filepaths))

    # make summary file
    summary_df = pd.DataFrame({
        'filename': filenames,
        'module_name': list(map(lambda module: module.__name__, modules)),
        'module_params': module_params,
        'execution_time': average_times,
    })

    # Run evaluation when there are more than one module.
    if len(modules) > 1:
        # pop general keys from strategies (e.g. metrics, speed_threshold)
        general_key = ['metrics', 'speed_threshold']
        general_strategy = dict(filter(lambda x: x[0] in general_key, strategies.items()))
        extra_strategy = dict(filter(lambda x: x[0] not in general_key, strategies.items()))

        # first, filter by threshold if it is enabled.
        if general_strategy.get('speed_threshold') is not None:
            results, filenames = filter_by_threshold(results, average_times, general_strategy['speed_threshold'],
                                                     filenames)

        # check metrics in strategy
        if general_strategy.get('metrics') is None:
            raise ValueError("You must at least one metrics for query expansion evaluation.")

        # get retrieval modules from strategy
        retrieval_callables, retrieval_params = make_retrieval_callable_params(extra_strategy)

        # get retrieval_gt
        retrieval_gt = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))['retrieval_gt'].tolist()

        # run evaluation
        evaluation_results = list(map(lambda result: evaluate_one_query_expansion_node(
            retrieval_callables, retrieval_params, result['queries'].tolist(), retrieval_gt,
            general_strategy['metrics'], project_dir, previous_result), results))

        for metric_name in general_strategy['metrics']:
            summary_df[f'query_expansion_{metric_name}'] = list(map(lambda x: x[metric_name].mean(), evaluation_results))

        best_result, best_filename = select_best_average(evaluation_results, general_strategy['metrics'], filenames)
        # change metric name columns to query_expansion_metric_name
        best_result = best_result.rename(columns={
            metric_name: f'query_expansion_{metric_name}' for metric_name in strategies['metrics']})
        best_result = best_result.drop(columns=['generated_texts'])
    else:
        best_result, best_filename = results[0], filenames[0]

    # add 'is_best' column at summary file
    summary_df['is_best'] = summary_df['filename'] == best_filename

    best_result = pd.concat([previous_result, best_result], axis=1)

    # save files
    summary_df.to_parquet(os.path.join(node_dir, "summary.parquet"), index=False)
    best_result.to_parquet(os.path.join(node_dir, f"best_{os.path.splitext(best_filename)[0]}.parquet"), index=False)

    return best_result


def evaluate_one_query_expansion_node(retrieval_funcs: List[Callable],
                                      retrieval_params: List[Dict],
                                      expanded_queries: List[List[str]],
                                      retrieval_gt: List[List[str]],
                                      metrics: List[str],
                                      project_dir,
                                      previous_result: pd.DataFrame) -> pd.DataFrame:
    previous_result['queries'] = expanded_queries
    retrieval_results = list(map(lambda x: x[0](project_dir=project_dir, previous_result=previous_result, **x[1]),
                                 zip(retrieval_funcs, retrieval_params)))
    evaluation_results = list(map(lambda x: evaluate_retrieval_node(x, retrieval_gt, metrics),
                                  retrieval_results))
    best_result, _ = select_best_average(evaluation_results, metrics)
    best_result = pd.concat([previous_result, best_result], axis=1)
    return best_result


def make_retrieval_callable_params(strategy_dict: Dict):
    """
    [example]
    strategies = {
            "metrics": ["retrieval_f1", "retrieval_recall"],
            "top_k": 50,
            "retrieval_modules": [
              {"module_type": "bm25"},
              {"module_type": "vectordb", "embedding_model": ["openai", "huggingface"]}
            ]
          }
    """
    node_dict = deepcopy(strategy_dict)
    retrieval_module_list: Optional[List[Dict]] = node_dict.pop('retrieval_modules', None)
    if retrieval_module_list is None:
        retrieval_module_list = [{
            'module_type': 'bm25',
        }]
    node_params = node_dict
    modules = list(map(lambda module_dict: get_support_modules(module_dict.pop('module_type')),
                       retrieval_module_list))
    param_combinations = list(map(lambda module_dict: make_combinations({**module_dict, **node_params}),
                                  retrieval_module_list))
    return explode(modules, param_combinations)
