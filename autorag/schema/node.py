import itertools
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Callable

import pandas as pd

from autorag.nodes.retrieval.run import run_retrieval_node
from autorag.schema.module import Module

SUPPORT_NODES = {
    'retrieval': run_retrieval_node,
}


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
        return self.run_node(modules=list(map(lambda x: x.module, self.modules)),
                             module_params=self.get_param_combinations(),
                             previous_result=previous_result,
                             node_line_dir=node_line_dir,
                             strategies=self.strategy)
