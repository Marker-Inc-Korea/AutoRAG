import logging
import os
import pathlib
from typing import List, Callable, Dict

import pandas as pd

from autorag.evaluate import evaluate_retrieval
from autorag.strategy import measure_speed, filter_by_threshold, select_best_average
from autorag.utils.util import make_module_file_name

logger = logging.getLogger("AutoRAG")


def run_passage_reranker_node(modules: List[Callable],
                       module_params: List[Dict],
                       previous_result: pd.DataFrame,
                       node_line_dir: str,
                       strategies: Dict,
                       ) -> pd.DataFrame:
    if not os.path.exists(node_line_dir):
        os.makedirs(node_line_dir)
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    retrieval_gt = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))['retrieval_gt'].tolist()

    results, execution_times = zip(*map(lambda task: measure_speed(
        task[0], project_dir=project_dir, previous_result=previous_result, **task[1]), zip(modules, module_params)))
    average_times = list(map(lambda x: x / len(results[0]), execution_times))



