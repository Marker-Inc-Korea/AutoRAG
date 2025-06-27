import os
import pathlib
from typing import List, Dict

import pandas as pd

from autorag.schema.metricinput import MetricInput
from autorag.utils.util import apply_recursive, to_list


def run_hybrid_retrieval_node(
	modules: List,
	module_params: List[Dict],
	previous_result: pd.DataFrame,
	node_line_dir: str,
	strategies: Dict,
) -> pd.DataFrame:
	"""
	Run the hybrid retrieval node with the given modules and parameters.

	:param modules: List of hybrid retrieval modules to run.
	:param module_params: List of parameters for each module.
	:param previous_result: DataFrame containing the previous results.
	:param node_line_dir: Directory where the results will be saved.
	:param strategies: Dictionary containing strategies for filtering and evaluation.
	:return: DataFrame containing the results of the hybrid retrieval.
	"""
	if not os.path.exists(node_line_dir):
		os.makedirs(node_line_dir)
	project_dir = pathlib.PurePath(node_line_dir).parent.parent
	qa_df = pd.read_parquet(
		os.path.join(project_dir, "data", "qa.parquet"), engine="pyarrow"
	)
	retrieval_gt = qa_df["retrieval_gt"].tolist()
	retrieval_gt = apply_recursive(lambda x: str(x), to_list(retrieval_gt))

	# make rows to metric_inputs
	_ = [
		MetricInput(retrieval_gt=ret_gt, query=query, generation_gt=gen_gt)
		for ret_gt, query, gen_gt in zip(
			retrieval_gt, qa_df["query"].tolist(), qa_df["generation_gt"].tolist()
		)
	]

	save_dir = os.path.join(node_line_dir, "hybrid_retrieval")  # node name
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
