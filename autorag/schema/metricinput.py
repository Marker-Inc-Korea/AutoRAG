from dataclasses import dataclass
from typing import Optional, List, Dict, Callable, Any


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

    def is_fields_notnone(self, fields_to_check: List[str]) -> bool:
        type_checks: Dict[type, Callable[[Any], bool]] = {
            str: lambda x: len(x.strip()) > 0,
            list: self._check_list,
            int: lambda _: True,
            float: lambda _: True,
        }
        for field in fields_to_check:
            actual_value = getattr(self, field)

            if actual_value is None:
                return False

            try:
                if not type_checks.get(type(actual_value), lambda _: False)(actual_value):
                    return False
            except Exception:
                return False

        return True

    def _check_list(self, lst: List[Any]) -> bool:
        if len(lst) == 0:
            return False

        for item in lst:
            if isinstance(item, str):
                if len(item.strip()) == 0:
                    return False
            elif isinstance(item, list):
                if not self._check_list(item):
                    return False
            elif item is None:
                return False

        return True
