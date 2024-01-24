import itertools
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Callable

import pandas as pd

from autorag.nodes.retrieval.run import run_retrieval_node
from autorag.schema.module import Module

SUPPORT_NODES = {
    'retrieval': run_retrieval_node,
}
logger = logging.getLogger("AutoRAG")


@dataclass
class Node:
    node_type: str
    strategy: Dict
    node_params: Dict
    modules: List[Module]
    run_node: Callable = field(init=False)

    def __post_init__(self):
        self.run_node = SUPPORT_NODES.get(self.node_type)
        if self.run_node is None:
            raise ValueError(f"Node type {self.node_type} is not supported.")

    def get_param_combinations(self) -> List[Dict]:
        """
        This method returns a combination of module and node parameters.

        :return: Module Parameters from each module.
        """
        input_dict = list(map(lambda x: {**self.node_params, **x.module_param}, self.modules))[0]

        dict_with_lists = {k: (v if isinstance(v, list) else [v]) for k, v in input_dict.items()}

        # Generate all combinations of values
        combinations = list(itertools.product(*dict_with_lists.values()))

        # Convert combinations back into dictionaries with the original keys
        combination_dicts = [dict(zip(dict_with_lists.keys(), combo)) for combo in combinations]
        return combination_dicts

    @classmethod
    def from_dict(cls, node_dict: Dict) -> 'Node':
        _node_dict = deepcopy(node_dict)
        node_type = _node_dict.pop('node_type')
        strategy = _node_dict.pop('strategy')
        modules = list(map(lambda x: Module.from_dict(x), _node_dict.pop('modules')))
        node_params = _node_dict
        return cls(node_type, strategy, node_params, modules)

    def run(self, previous_result: pd.DataFrame, node_line_dir: str) -> pd.DataFrame:
        logger.info(f'Running node {self.node_type}...')
        return self.run_node(modules=list(map(lambda x: x.module, self.modules)),
                             module_params=self.get_param_combinations(),
                             previous_result=previous_result,
                             node_line_dir=node_line_dir,
                             strategies=self.strategy)


def extract_values(node: Node, key: str) -> List[str]:
    """
    This function extract values from node's modules' module_param.

    :param node: The node you want to extract values from.
    :param key: The key of module_param that you want to extract.
    :return: The list of extracted values.
        It removes duplicated elements automatically.
    """

    def extract_module_values(module: Module):
        if key not in module.module_param:
            return []
        value = module.module_param[key]
        if isinstance(value, str):
            return [value]
        elif isinstance(value, list):
            return value
        else:
            raise ValueError(f"{key} must be str or list, but got {type(value)}")

    values = list(map(extract_module_values, node.modules))
    return list(set(list(itertools.chain.from_iterable(values))))


def extract_values_from_nodes(nodes: List[Node], key: str) -> List[str]:
    """
    This function extract values from nodes' modules' module_param.

    :param nodes: The nodes you want to extract values from.
    :param key: The key of module_param that you want to extract.
    :return: The list of extracted values.
        It removes duplicated elements automatically.
    """
    values = list(map(lambda node: extract_values(node, key), nodes))
    return list(set(list(itertools.chain.from_iterable(values))))


def module_type_exists(nodes: List[Node], module_type: str) -> bool:
    """
    This function check if the module type exists in the nodes.

    :param nodes: The nodes you want to check.
    :param module_type: The module type you want to check.
    :return: True if the module type exists in the nodes.
    """
    return any(list(map(lambda node: any(list(map(lambda module: module.module_type == module_type, node.modules))),
                        nodes)))
