import asyncio
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from autorag.nodes.passagereranker.base import passage_reranker_node


@passage_reranker_node
def koreranker(queries: List[str], contents_list: List[List[str]],
               scores_list: List[List[float]], ids_list: List[List[str]],
               top_k: int) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    model_path = "Dongjin-kr/ko-reranker"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run async ko_rerank_pure function
    tasks = [koreranker_pure(query, contents, scores, ids, top_k, model, tokenizer, device)
             for query, contents, scores, ids in zip(queries, contents_list, scores_list, ids_list)]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    content_result = list(map(lambda x: x[0], results))
    id_result = list(map(lambda x: x[1], results))
    score_result = list(map(lambda x: x[2], results))
    return content_result, id_result, score_result


async def koreranker_pure(query: str, contents: List[str],
                          scores: List[float], ids: List[str],
                          top_k: int, model, tokenizer, device) \
        -> Tuple[List[str], List[str], List[float]]:
    model.to(device)

    input_pairs = [[query, content] for content in contents]
    with torch.no_grad():
        inputs = tokenizer(input_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = exp_normalize(scores.numpy())

    # Convert the scores to float
    scores = scores.astype(float)

    # Create a list of tuples pairing each content with its relevance probability
    content_ids_scores = list(zip(contents, ids, scores))

    # Sort the list of pairs based on the relevance score in descending order
    sorted_content_ids_scores = sorted(content_ids_scores, key=lambda x: x[2], reverse=True)

    # crop with top_k
    if len(contents) < top_k:
        top_k = len(contents)
    sorted_content_ids_scores = sorted_content_ids_scores[:top_k]

    content_result, id_result, score_result = zip(*sorted_content_ids_scores)

    return list(content_result), list(id_result), list(score_result)


def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()
