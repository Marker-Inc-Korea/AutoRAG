import os
import pathlib
from copy import deepcopy
from typing import List, Callable, Dict, Optional, Union

import pandas as pd

from autorag.evaluate import evaluate_generation
from autorag.evaluate.util import cast_metrics
from autorag.strategy import measure_speed, filter_by_threshold, select_best_average
from autorag.support import get_support_modules
from autorag.utils import validate_qa_dataset
from autorag.utils.util import make_combinations, explode


def run_prompt_maker_node(modules: List[Callable],
                          module_params: List[Dict],
                          previous_result: pd.DataFrame,
                          node_line_dir: str,
                          strategies: Dict,
                          ) -> pd.DataFrame:
    """
    Run prompt maker node.
    With this function, you can select the best prompt maker module.
    As default, when you can use only one module, the evaluation will be skipped.
    If you want to select the best prompt among modules, you can use strategies.
    When you use them, you must pass 'generator_modules' and its parameters at strategies.
    Because it uses generator modules and generator metrics for evaluation this module.
    It is recommended to use one params and modules for evaluation,
    but you can use multiple params and modules for evaluation.
    When you don't set generator module at strategies, it will use the default generator module.
    The default generator module is llama_index_llm with openai gpt-3.5-turbo model.

    :param modules: Prompt maker modules to run.
    :param module_params: Prompt maker module parameters.
    :param previous_result: Previous result dataframe.
        Could be query expansion's best result or qa data.
    :param node_line_dir: This node line's directory.
    :param strategies: Strategies for prompt maker node.
    :return: The best result dataframe.
        It contains previous result columns and prompt maker's result columns which is 'prompts'.
    """
    if not os.path.exists(node_line_dir):
        os.makedirs(node_line_dir)
    node_dir = os.path.join(node_line_dir, "prompt_maker")
    if not os.path.exists(node_dir):
        os.makedirs(node_dir)
    project_dir = pathlib.PurePath(node_line_dir).parent.parent

    # run modules
    results, execution_times = zip(*map(lambda task: measure_speed(
        task[0], project_dir=project_dir, previous_result=previous_result, **task[1]), zip(modules, module_params)))
    average_times = list(map(lambda x: x / len(results[0]), execution_times))

    # save results to folder
    filepaths = list(map(lambda x: os.path.join(node_dir, f'{x}.parquet'), range(len(modules))))
    list(map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths)))  # execute save to parquet
    filenames = list(map(lambda x: os.path.basename(x), filepaths))

    # make summary file
    summary_df = pd.DataFrame({
        'filename': filenames,
        'module_name': list(map(lambda module: module.__name__, modules)),
        'module_params': module_params,
        'execution_time': average_times,
    })

    metric_names, metric_params = cast_metrics(strategies.get('metrics'))

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

        # run metrics before filtering
        if metric_names is None or len(metric_names) <= 0:
            raise ValueError("You must at least one metrics for prompt maker evaluation.")

        # get generator modules from strategy
        generator_callables, generator_params = make_generator_callable_params(extra_strategy)

        # get generation_gt
        qa_data = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))
        validate_qa_dataset(qa_data)
        generation_gt = qa_data['generation_gt'].tolist()
        generation_gt = list(map(lambda x: x.tolist(), generation_gt))

        # run evaluations
        evaluation_results = list(map(lambda result: evaluate_one_prompt_maker_node(
            generator_callables, generator_params, result['prompts'].tolist(),
            generation_gt, general_strategy['metrics'], project_dir), results))

        evaluation_df = pd.DataFrame({
            'filename': filenames,
            **{f'prompt_maker_{metric_name}': list(map(lambda x: x[metric_name].mean(), evaluation_results))
               for metric_name in metric_names}
        })
        summary_df = pd.merge(on='filename', left=summary_df, right=evaluation_df, how='left')

        best_result, best_filename = select_best_average(evaluation_results, metric_names, filenames)
        # change metric name columns to prompt_maker_metric_name
        best_result = best_result.rename(columns={
            metric_name: f'prompt_maker_{metric_name}' for metric_name in metric_names})
        best_result = best_result.drop(columns=['generated_texts'])
    else:
        best_result, best_filename = results[0], filenames[0]

    # add 'is_best' column at summary file
    summary_df['is_best'] = summary_df['filename'] == best_filename

    best_result = pd.concat([previous_result, best_result], axis=1)

    # save files
    summary_df.to_csv(os.path.join(node_dir, "summary.csv"), index=False)
    best_result.to_parquet(os.path.join(node_dir, f"best_{os.path.splitext(best_filename)[0]}.parquet"), index=False)

    return best_result


def make_generator_callable_params(strategy_dict: Dict):
    node_dict = deepcopy(strategy_dict)
    generator_module_list: Optional[List[Dict]] = node_dict.pop('generator_modules', None)
    if generator_module_list is None:
        generator_module_list = [{
            'module_type': 'llama_index_llm',
            'llm': 'openai',
            'model': 'gpt-3.5-turbo',
        }]
    node_params = node_dict
    modules = list(map(lambda module_dict: get_support_modules(module_dict.pop('module_type')),
                       generator_module_list))
    param_combinations = list(map(lambda module_dict: make_combinations({**module_dict, **node_params}),
                                  generator_module_list))
    return explode(modules, param_combinations)


def evaluate_one_prompt_maker_node(generator_funcs: List[Callable],
                                   generator_params: List[Dict],
                                   prompts: List[str],
                                   generation_gt: List[List[str]],
                                   metrics: Union[List[str], List[Dict]],
                                   project_dir) -> pd.DataFrame:
    input_df = pd.DataFrame({'prompts': prompts})
    generator_results = list(map(lambda x: x[0](project_dir=project_dir, previous_result=input_df, **x[1]),
                                 zip(generator_funcs, generator_params)))
    evaluation_results = list(map(lambda x: evaluate_generator_result(x[0], generation_gt, metrics),
                                  zip(generator_results, generator_funcs)))
    metric_names = list(map(lambda x: x['metric_name'], metrics)) if isinstance(metrics[0], dict) else metrics
    best_result, _ = select_best_average(evaluation_results, metric_names)
    best_result = pd.concat([input_df, best_result], axis=1)
    return best_result  # it has 'generated_texts' column


def evaluate_generator_result(result_df: pd.DataFrame,
                              generation_gt: List[List[str]],
                              metrics: Union[List[str], List[Dict]]) -> pd.DataFrame:
    @evaluate_generation(generation_gt=generation_gt, metrics=metrics)
    def evaluate(df):
        return df['generated_texts'].tolist()

    return evaluate(result_df)
