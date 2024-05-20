import os
import pathlib
from typing import List, Callable, Dict

import pandas as pd

from autorag.nodes.retrieval.run import evaluate_retrieval_node
from autorag.strategy import measure_speed, filter_by_threshold, select_best


def run_passage_filter_node(modules: List[Callable],
                            module_params: List[Dict],
                            previous_result: pd.DataFrame,
                            node_line_dir: str,
                            strategies: Dict) -> pd.DataFrame:
    """
    Run evaluation and select the best module among passage filter node results.

    :param modules: Passage filter modules to run.
    :param module_params: Passage filter module parameters.
    :param previous_result: Previous result dataframe.
        Could be retrieval, reranker, passage filter modules result.
        It means it must contain 'query', 'retrieved_contents', 'retrieved_ids', 'retrieve_scores' columns.
    :param node_line_dir: This node line's directory.
    :param strategies: Strategies for passage filter node.
        In this node, we use 'retrieval_f1', 'retrieval_recall' and 'retrieval_precision'.
        You can skip evaluation when you use only one module and a module parameter.
    :return: The best result dataframe with previous result columns.
    """
    if not os.path.exists(node_line_dir):
        os.makedirs(node_line_dir)
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    qa_df = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))
    retrieval_gt = qa_df['retrieval_gt'].tolist()
    retrieval_gt = [[[str(uuid) for uuid in sub_array] if sub_array.size > 0 else [] for sub_array in inner_array]
                    for inner_array in retrieval_gt]

    results, execution_times = zip(*map(lambda task: measure_speed(
        task[0], project_dir=project_dir, previous_result=previous_result, **task[1]), zip(modules, module_params)))
    average_times = list(map(lambda x: x / len(results[0]), execution_times))

    # run metrics before filtering
    if strategies.get('metrics') is None:
        raise ValueError("You must at least one metrics for passage_filter evaluation.")
    results = list(map(lambda x: evaluate_retrieval_node(x, retrieval_gt, strategies.get('metrics'),
                                                         qa_df['query'].tolist(),
                                                         qa_df['generation_gt'].tolist()), results))

    # save results to folder
    save_dir = os.path.join(node_line_dir, "passage_filter")  # node name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepaths = list(map(lambda x: os.path.join(save_dir, f'{x}.parquet'), range(len(modules))))
    list(map(lambda x: x[0].to_parquet(x[1], index=False), zip(results, filepaths)))  # execute save to parquet
    filenames = list(map(lambda x: os.path.basename(x), filepaths))

    summary_df = pd.DataFrame({
        'filename': filenames,
        'module_name': list(map(lambda module: module.__name__, modules)),
        'module_params': module_params,
        'execution_time': average_times,
        **{f'passage_filter_{metric}': list(map(lambda result: result[metric].mean(), results)) for metric in
           strategies.get('metrics')},
    })

    # filter by strategies
    if strategies.get('speed_threshold') is not None:
        results, filenames = filter_by_threshold(results, average_times, strategies['speed_threshold'], filenames)
    selected_result, selected_filename = select_best(results, strategies.get('metrics'), filenames,
                                                     strategies.get('strategy', 'mean'))
    selected_result = selected_result.rename(columns={
        metric_name: f'passage_filter_{metric_name}' for metric_name in strategies['metrics']})
    previous_result = previous_result.drop(columns=['retrieved_contents', 'retrieved_ids', 'retrieve_scores'])
    best_result = pd.concat([previous_result, selected_result], axis=1)

    # add 'is_best' column to summary file
    summary_df['is_best'] = summary_df['filename'] == selected_filename

    # save files
    summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
    best_result.to_parquet(os.path.join(save_dir, f'best_{os.path.splitext(selected_filename)[0]}.parquet'),
                           index=False)
    return best_result
