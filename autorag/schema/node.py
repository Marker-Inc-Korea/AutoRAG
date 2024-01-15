from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Callable

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

    def get_module_node_params(self) -> List[Dict]:
        """
        This method returns module parameters for each module, including node parameters.

        :return: Module Parameters from each module.
        """
        return list(map(lambda x: {**self.node_params, **x.module_param}, self.modules))

    @classmethod
    def from_dict(cls, node_dict: Dict) -> 'Node':
        _node_dict = deepcopy(node_dict)
        node_type = _node_dict.pop('node_type')
        strategy = _node_dict.pop('strategy')
        modules = list(map(lambda x: Module.from_dict(x), _node_dict.pop('modules')))
        node_params = _node_dict
        return cls(node_type, strategy, node_params, modules)
