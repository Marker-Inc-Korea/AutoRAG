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
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    resources_dir = os.path.join(project_dir, "resources")
    data_dir = os.path.join(project_dir, "data")
    retrieval_gt = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))['retrieval_gt'].tolist()

    # run query expansion
    result_queries, execution_times = zip(*map(lambda task: measure_speed(
        task[0], project_dir=project_dir, previous_result=previous_result, **task[1]), zip(modules, module_params)))
    average_times = list(map(lambda x: x / len(result_queries[0]), execution_times))

    # get retrieval module values
    if "retrieval_modules" in strategies.keys():
        retrieval_modules = strategies.pop("retrieval_modules")
    else:
        retrieval_modules = [{"module_type": "bm25"}]  # default value

    # get the best retrieval result for each module
    results = list(map(lambda x: module_best_retrieval(x, retrieval_modules, project_dir,
                                                       retrieval_gt, strategies, previous_result), result_queries))

    # save results to folder
    save_dir = os.path.join(node_line_dir, "query_expansion")  # node name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepaths = list(map(lambda x: os.path.join(save_dir, make_module_file_name(x[0].__name__, x[1])),
                         zip(modules, module_params)))
    list(map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths)))  # execute save to parquet
    filenames = list(map(lambda x: os.path.basename(x), filepaths))

    summary_df = pd.DataFrame({
        'filename': filenames,
        'module_name': list(map(lambda module: module.__name__, modules)),
        'module_params': module_params,
        'execution_time': average_times,
        **{f"query_expansion_{metric}": list(map(lambda result: result[metric].mean(), results)) for metric in strategies.get('metrics')},
    })

    # filter by strategies
    if strategies.get('speed_threshold') is not None:
        results, filenames = filter_by_threshold(results, average_times, strategies['speed_threshold'], filenames)
    selected_result, selected_filename = select_best_average(results, strategies.get('metrics'), filenames)
    best_result = pd.concat([previous_result, selected_result], axis=1)

    # add 'is_best' column at summary file
    summary_df['is_best'] = summary_df['filename'] == selected_filename

    # save the result files
    best_result.to_parquet(os.path.join(save_dir, f'best_{os.path.splitext(selected_filename)[0]}.parquet'),
                           index=False)
    summary_df.to_parquet(os.path.join(save_dir, 'summary.parquet'), index=False)
    return best_result


def process_retrieval_module(retrieval_module, project_dir, expanded_queries, top_k, previous_result):
    retrieval_module_type = retrieval_module.pop('module_type')

    previous_result['queries'] = expanded_queries

    retrieval_callabes, 

    if retrieval_module_type == "bm25":
        result_df = bm25(project_dir=project_dir, previous_result=previous_result, top_k=top_k)
    elif retrieval_module_type == "vectordb":
        embedding_model_str = retrieval_module.pop("embedding_model")
        result_df = vectordb(project_dir=project_dir, previous_result=previous_result, top_k=top_k, embedding_model=embedding_model_str)
    else:
        raise ValueError(f"invalid f{retrieval_module_type} for using query_expansion_io decorator.")

    contents = result_df["retrieved_contents"].tolist()
    ids = result_df["retrieved_ids"].tolist()
    scores = result_df["retrieve_scores"].tolist()

    # create retrieval df
    retrieval_df = pd.DataFrame({
        'retrieved_contents': contents,
        'retrieved_ids': ids,
        'retrieve_scores': scores,
    })
    return retrieval_df


def make_retrieval_callable_params(retrieval_module: Dict):
    """
    [example]
    retrieval_modules = [
    {"module_type": "bm25"},
    {"module_type": "vectordb", "embedding_model": ["openai", huggingface"]}
    ]
    """
    retrieval_dict = deepcopy(retrieval_module)
    generator_module_list: Optional[List[Dict]] = retrieval_dict.pop('generator_modules', None)
    if generator_module_list is None:
        generator_module_list = [{
            'module_type': 'llama_index_llm',
            'llm': 'openai',
            'model_name': 'gpt-3.5-turbo',
        }]
    node_params = retrieval_dict
    modules = list(map(lambda module_dict: get_support_modules(module_dict.pop('module_type')),
                       generator_module_list))
    param_combinations = list(map(lambda module_dict: make_combinations({**module_dict, **node_params}),
                                  generator_module_list))
    return explode(modules, param_combinations)


def module_best_retrieval(result_df: pd.DataFrame, retrieval_modules: List[Dict], project_dir,
                          retrieval_gt, strategies, previous_result: pd.DataFrame):
    # get expanded_queries to list
    expanded_queries = result_df["queries"].tolist()

    # get all combinations of retrieval modules
    final_retrieval_modules = [item for retrieval_module in retrieval_modules
                               for item in make_combinations(retrieval_module)]

    # get retrieval results
    if strategies.get('top_k') is None:
        raise ValueError("You must specify top_k for retrieval evaluation.")
    retrieval_results = list(map(lambda x: process_retrieval_module(x, project_dir, expanded_queries,
                                                                    strategies.get('top_k'), previous_result), final_retrieval_modules))
    # get best retrieval result for each retrieval module
    if strategies.get('metrics') is None:
        raise ValueError("You must at least one metrics for retrieval evaluation.")
    results = list(map(lambda x: evaluate_retrieval_node(x, retrieval_gt, strategies.get('metrics')), retrieval_results))

    # filter best result in retrieval_modules
    best_result = select_best_average(results, strategies.get('metrics'))
    return best_result[0]
