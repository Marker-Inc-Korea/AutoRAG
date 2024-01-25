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

    # 질문 리스트
    # 1. query_expansion 에서 llm 선택할 수 있게 해야?
    # 2. query_expansion metric을 어떻게 찍어낼 것인가?
    # -> retrieval default: bm25, retrieval module은 yaml에서 받아옴
    # -> 예상 yaml파일 적어보기
