from dataclasses import dataclass
from typing import Optional, List, Dict, Callable, Any, Union

import numpy as np
import pandas as pd


@dataclass
class MetricInput:
    query: Optional[str] = None
    queries: Optional[List[str]] = None
    gt_contents: Optional[List[str]] = None
    retrieval_contents: Optional[List[str]] = None
    retrieval_gt: Optional[List[List[str]]] = None
    retrieval_ids: Optional[List[str]] = None
    prompt: Optional[str] = None
    generated_texts: Optional[str] = None
    generation_gt: Optional[List[str]] = None
    generated_log_probs: Optional[List[float]] = None

    def is_fields_notnone(cls, fields_to_check: List[str]) -> bool:
        type_checks: Dict[type, Callable[[Any], bool]] = {
            str: lambda x: len(x.strip()) > 0,
            list: cls._check_list,
            int: lambda _: True,
            float: lambda _: True,
            np.ndarray: cls._check_list
        }
        for field in fields_to_check:
            actual_value = getattr(cls, field)

            if actual_value is None:
                return False

            try:
                if not type_checks.get(type(actual_value), lambda _: False)(actual_value):
                    return False
            except Exception:
                return False

        return True

    @classmethod
    def from_dataframe(cls, qa_data: pd.DataFrame) -> List['MetricInput']:
        """
        Convert a pandas DataFrame into a list of MetricInput instances.
        qa_data: pd.DataFrame: qa_data DataFrame containing metric data.

        :returns: List[MetricInput]: List of MetricInput objects created from DataFrame rows.
        """
        instances = []

        for _, row in qa_data.iterrows():
            instance = cls()

            for attr_name in cls.__annotations__:
                if attr_name in row:
                    value = row[attr_name]

                    if isinstance(value, str):
                        setattr(instance, attr_name, value.strip() if value.strip() != '' else None)
                    elif isinstance(value, list):
                        setattr(instance, attr_name, value if len(value) > 0 else None)
                    elif isinstance(value, (int, float)):
                        setattr(instance, attr_name, value)
                    else:
                        setattr(instance, attr_name, value)

            instances.append(instance)

        return instances

    def _check_list(self, lst_or_arr: Union[List[Any], np.ndarray]) -> bool:
        if isinstance(lst_or_arr, np.ndarray):
            lst_or_arr = lst_or_arr.flatten().tolist()

        if len(lst_or_arr) == 0:
            return False

        type_checks: Dict[type, Callable[[Any], bool]] = {
            str: lambda x: len(x.strip()) > 0,
            list: self._check_list,
            np.ndarray: self._check_list,
            int: lambda _: True,
            float: lambda _: True,
        }

        for item in lst_or_arr:
            if item is None:
                return False

            item_type = type(item)

            if item_type in type_checks:
                if not type_checks[item_type](item):
                    return False
            else:
                return False

        return True
