import itertools
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Tuple, Any

import pandas as pd

from autorag.schema.module import Module
from autorag.support import get_support_nodes
from autorag.utils.util import make_combinations, explode, find_key_values

logger = logging.getLogger("AutoRAG")


@dataclass
class Node:
	node_type: str
	strategy: Dict
	node_params: Dict
	modules: List[Module]
	run_node: Callable = field(init=False)

	def __post_init__(self):
		self.run_node = get_support_nodes(self.node_type)
		if self.run_node is None:
			raise ValueError(f"Node type {self.node_type} is not supported.")

	def get_param_combinations(self) -> Tuple[List[Callable], List[Dict]]:
		"""
		This method returns a combination of module and node parameters, also corresponding modules.

		:return: Each module and its module parameters.
		:rtype: Tuple[List[Callable], List[Dict]]
		"""

		def make_single_combination(module: Module) -> List[Dict]:
			input_dict = {**self.node_params, **module.module_param}
			return make_combinations(input_dict)

		combinations = list(map(make_single_combination, self.modules))
		module_list, combination_list = explode(self.modules, combinations)
		return list(map(lambda x: x.module, module_list)), combination_list

	@classmethod
	def from_dict(cls, node_dict: Dict) -> "Node":
		_node_dict = deepcopy(node_dict)
		node_type = _node_dict.pop("node_type")
		strategy = _node_dict.pop("strategy")
		modules = list(map(lambda x: Module.from_dict(x), _node_dict.pop("modules")))
		node_params = _node_dict
		return cls(node_type, strategy, node_params, modules)

	def run(self, previous_result: pd.DataFrame, node_line_dir: str) -> pd.DataFrame:
		logger.info(f"Running node {self.node_type}...")
		input_modules, input_params = self.get_param_combinations()
		return self.run_node(
			modules=input_modules,
			module_params=input_params,
			previous_result=previous_result,
			node_line_dir=node_line_dir,
			strategies=self.strategy,
		)


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
		if isinstance(value, str) or isinstance(value, int):
			return [value]
		elif isinstance(value, list):
			return value
		else:
			raise ValueError(f"{key} must be str,list or int, but got {type(value)}")

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


def extract_values_from_nodes_strategy(nodes: List[Node], key: str) -> List[Any]:
	"""
	This function extract values from nodes' strategy.

	:param nodes: The nodes you want to extract values from.
	:param key: The key string that you want to extract.
	:return: The list of extracted values.
	    It removes duplicated elements automatically.
	"""
	values = []
	for node in nodes:
		value_list = find_key_values(node.strategy, key)
		if value_list:
			values.extend(value_list)
	return values


def module_type_exists(nodes: List[Node], module_type: str) -> bool:
	"""
	This function check if the module type exists in the nodes.

	:param nodes: The nodes you want to check.
	:param module_type: The module type you want to check.
	:return: True if the module type exists in the nodes.
	"""
	return any(
		list(
			map(
				lambda node: any(
					list(
						map(
							lambda module: module.module_type == module_type,
							node.modules,
						)
					)
				),
				nodes,
			)
		)
	)
