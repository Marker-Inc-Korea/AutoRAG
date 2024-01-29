import os.path
import pathlib
from typing import Callable, List, Dict

import pandas as pd

from autorag.strategy import measure_speed


def run_passage_compressor_node(modules: List[Callable],
                                module_params: List[Dict],
                                previous_result: pd.DataFrame,
                                node_line_dir: str,
                                strategies: Dict,
                                ) -> pd.DataFrame:
    """
    Run evaluation and select the best module among passage compressor modules.

    :param modules: Passage compressor modules to run.
    :param module_params: Passage compressor module parameters.
    :param previous_result: Previous result dataframe.
        Could be retrieval, reranker modules result.
        It means it must contain 'query', 'retrieved_contents', 'retrieved_ids', 'retrieve_scores' columns.
    :param node_line_dir: This node line's directory.
    :param strategies: Strategies for passage compressor node.
        In this node, we use generation metrics to evaluate.
        So it is recommended to add generation_modules at strategies.
        When you don't, we use default generation module, which is openai gpt-3.5-turbo.
        You can skip evaluation when you use only one module and a module parameter.
    :return: The best result dataframe with previous result columns.
        This node will replace 'retrieved_contents' to compressed passages, so its length will be one.
    """
    if not os.path.exists(node_line_dir):
        os.makedirs(node_line_dir)
    project_dir = pathlib.PurePath(node_line_dir).parent.parent

    # run modules
    results, execution_times = zip(*map(lambda task: measure_speed(
        task[0], project_dir=project_dir, previous_result=previous_result, **task[1]), zip(modules, module_params)))
