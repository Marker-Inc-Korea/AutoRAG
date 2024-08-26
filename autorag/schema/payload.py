from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Payload:
    query: Optional[str] = None
    queries: Optional[List[str]] = None
    passage_contents: Optional[List[str]] = None
    passage_ids: Optional[List[str]] = None
    prompt: Optional[str] = None
    generated_texts: Optional[str] = None
    generation_gt: Optional[str] = None
    generated_log_probs: Optional[List[float]] = None
