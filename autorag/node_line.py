import os
import pathlib
from typing import Dict, List, Optional

import pandas as pd

from autorag.schema import Node
from autorag.utils.util import find_best_result_path


def make_node_lines(node_line_dict: Dict) -> List[Node]:
    """
    This method makes a list of nodes from node line dictionary.
    :param node_line_dict: Node_line_dict loaded from yaml file, or get from user input.
    :return: List of Nodes inside this node line.
    """
    nodes = node_line_dict.get('nodes')
    if nodes is None:
        raise ValueError("Node line must have \'nodes\' key.")
    node_objects = list(map(lambda x: Node.from_dict(x), nodes))
    return node_objects


def run_node_line(nodes: List[Node],
                  node_line_dir: str,
                  previous_result: Optional[pd.DataFrame] = None):
    """
    Run the whole node line by running each node.

    :param nodes: A list of nodes.
    :param node_line_dir: This node line's directory.
    :param previous_result: A result of the previous node line.
        If None, it loads qa data from data/qa.parquet.
    :return: The final result of the node line.
    """
    if previous_result is None:
        project_dir = pathlib.PurePath(node_line_dir).parent.parent
        qa_path = os.path.join(project_dir, "data", "qa.parquet")
        if not os.path.exists(qa_path):
            raise ValueError(f"qa.parquet does not exist in {qa_path}.")
        previous_result = pd.read_parquet(qa_path)

    summary_lst = []
    for node in nodes:
        previous_result = node.run(previous_result, node_line_dir)
        best_module_filename = os.path.basename(find_best_result_path(os.path.join(node_line_dir, node.node_type)))
        summary_lst.append({'node_type': node.node_type, 'best_module_filename': best_module_filename})
    summary_df = pd.DataFrame(summary_lst)
    summary_df.to_csv(os.path.join(node_line_dir, 'summary.csv'), index=False)
    return previous_result
