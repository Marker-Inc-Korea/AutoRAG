import logging
import os
import pathlib
from typing import List, Callable, Dict

import pandas as pd

from autorag.evaluate import evaluate_retrieval
from autorag.strategy import measure_speed, filter_by_threshold, select_best_average
from autorag.utils.util import make_module_file_name

logger = logging.getLogger("AutoRAG")


def run_query_expansion_node(modules: List[Callable],
                             module_params: List[Dict],
                             previous_result: pd.DataFrame,
                             node_line_dir: str,
                             strategies: Dict,
                             ) -> pd.DataFrame:
    if not os.path.exists(node_line_dir):
        os.makedirs(node_line_dir)
    project_dir = pathlib.PurePath(node_line_dir).parent.parent
    retrieval_gt = pd.read_parquet(os.path.join(project_dir, "data", "qa.parquet"))['retrieval_gt'].tolist()

