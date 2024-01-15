import asyncio
from typing import List, Tuple
from uuid import UUID

import torch
import torch.nn.functional as F

from .modeling_enc_t5 import EncT5ForSequenceClassification
from .tokenization_enc_t5 import EncT5Tokenizer


def tart(queries: List[str], contents_list: List[List[str]],
         scores_list: List[List[float]], ids_list: List[List[UUID]],
         instruction: str = "Find passage to answer given question") -> List[Tuple[List[str]]]:
    model_name = "facebook/tart-full-flan-t5-xl"
    model = EncT5ForSequenceClassification.from_pretrained(model_name)
    tokenizer = EncT5Tokenizer.from_pretrained(model_name)
    # Run async tart_rerank_pure function
    tasks = [tart_pure(query, contents, scores, ids, model, tokenizer, instruction) \
             for query, contents, scores, ids in zip(queries, contents_list, scores_list, ids_list)]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    return results


async def tart_pure(query: str, contents: List[str], scores: List[float], ids: List[UUID],
                    model, tokenizer, instruction: str) -> Tuple[List[str]]:
    instruction_queries: List[str] = ['{0} [SEP] {1}'.format(instruction, query) for _ in range(len(contents))]
    features = tokenizer(instruction_queries, contents, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**features).logits
        normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]

    contents_ids_scores = list(zip(contents, ids, normalized_scores))

    sorted_contents_ids_scores = sorted(contents_ids_scores, key=lambda x: x[2], reverse=True)

    return tuple(map(list, zip(*sorted_contents_ids_scores)))
