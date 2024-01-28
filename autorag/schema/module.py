from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict

from autorag.nodes.queryexpansion import query_decompose, hyde
from autorag.nodes.retrieval import bm25, vectordb

SUPPORT_MODULES = {
    'query_decompose': query_decompose,
    'hyde': hyde,
    'bm25': bm25,
    'vectordb': vectordb,
}


@dataclass
class Module:
    module_type: str
    module_param: Dict
    module: Callable = field(init=False)

    def __post_init__(self):
        self.module = SUPPORT_MODULES.get(self.module_type)
        if self.module is None:
            raise ValueError(f"Module type {self.module_type} is not supported.")

    @classmethod
    def from_dict(cls, module_dict: Dict) -> 'Module':
        _module_dict = deepcopy(module_dict)
        module_type = _module_dict.pop('module_type')
        module_params = _module_dict
        return cls(module_type, module_params)
