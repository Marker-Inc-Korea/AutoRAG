from dataclasses import dataclass
from typing import Optional, List

from autorag.evaluation.metric.generation import (
    bleu,
    meteor,
    rouge,
    sem_score,
    g_eval,
    bert_score
)
from autorag.evaluation.metric.retrieval import (
    retrieval_recall,
    retrieval_precision,
    retrieval_f1,
    retrieval_ndcg,
    retrieval_mrr,
    retrieval_map,
)
from autorag.evaluation.metric.retrieval_contents import (
    retrieval_token_f1,
    retrieval_token_precision,
    retrieval_token_recall,
)


@dataclass
class Payload:
    query: Optional[str] = None
    queries: Optional[List[str]] = None
    gt_contents: Optional[List[str]] = None
    retrieval_gt: Optional[List[List[str]]] = None
    prompt: Optional[str] = None
    generated_texts: Optional[str] = None
    generation_gt: Optional[List[str]] = None
    generated_log_probs: Optional[List[float]] = None


# input dict of metric excluding evaluation target
METRIC_INPUT_DICT = {
    # for generation metric
    bleu.__name__: ["generation_gt"],
    meteor.__name__: ["generation_gt"],
    rouge.__name__: ["generation_gt"],
    sem_score.__name__: ["generation_gt"],
    g_eval.__name__: ["generation_gt"],
    bert_score.__name__: ["generation_gt"],
    # for retrieval metric
    retrieval_recall.__name__: ["retrieval_gt"],
    retrieval_precision.__name__: ["retrieval_gt"],
    retrieval_f1.__name__: ["retrieval_gt"],
    retrieval_ndcg.__name__: ["retrieval_gt"],
    retrieval_mrr.__name__: ["retrieval_gt"],
    retrieval_map.__name__: ["retrieval_gt"],
    # for retrieval_contents metric
    retrieval_token_f1.__name__: ["gt_contents"],
    retrieval_token_precision.__name__: ["gt_contents"],
    retrieval_token_recall.__name__: ["gt_contents"],
}
