from dataclasses import dataclass
from typing import Optional, List


@dataclass
class MetricInput:
    query: Optional[str] = None
    queries: Optional[List[str]] = None
    gt_contents: Optional[List[str]] = None
    retrieval_gt: Optional[List[List[str]]] = None
    prompt: Optional[str] = None
    generated_texts: Optional[str] = None
    generation_gt: Optional[List[str]] = None
    generated_log_probs: Optional[List[float]] = None

    def is_fields_notnone(self, fields_to_check: List[str]) -> bool:

        for field in fields_to_check:
            actual_value = getattr(self, field)

            if actual_value is None:
                return False
        return True
