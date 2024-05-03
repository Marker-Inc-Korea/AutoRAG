from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import make_batch, sort_by_scores, flatten_apply, select_top_k


@passage_reranker_node
def koreranker(queries: List[str], contents_list: List[List[str]],
               scores_list: List[List[float]], ids_list: List[List[str]],
               top_k: int, batch: int = 64) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Rerank a list of contents based on their relevance to a query using ko-reranker.
    ko-reranker is a reranker based on korean (https://huggingface.co/Dongjin-kr/ko-reranker).

    :param queries: The list of queries to use for reranking
    :param contents_list: The list of lists of contents to rerank
    :param scores_list: The list of lists of scores retrieved from the initial ranking
    :param ids_list: The list of lists of ids retrieved from the initial ranking
    :param top_k: The number of passages to be retrieved
    :param batch: The number of queries to be processed in a batch
        Default is 64.
    :return: tuple of lists containing the reranked contents, ids, and scores
    """
    model_path = "Dongjin-kr/ko-reranker"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    nested_list = [list(map(lambda x: [query, x], content_list)) for query, content_list in zip(queries, contents_list)]
    scores_nps = flatten_apply(koreranker_run_model, nested_list, model=model, batch_size=batch,
                               tokenizer=tokenizer, device=device)

    rerank_scores = list(map(lambda scores: exp_normalize(np.array(scores)).astype(float), scores_nps))

    df = pd.DataFrame({
        'contents': contents_list,
        'ids': ids_list,
        'scores': rerank_scores,
    })
    df[['contents', 'ids', 'scores']] = df.apply(sort_by_scores, axis=1, result_type='expand')
    results = select_top_k(df, ['contents', 'ids', 'scores'], top_k)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results['contents'].tolist(), results['ids'].tolist(), results['scores'].tolist()


def koreranker_run_model(input_texts, model, tokenizer, device, batch_size: int):
    batch_input_texts = make_batch(input_texts, batch_size)
    results = []
    for batch_texts in tqdm(batch_input_texts):
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = inputs.to(device)
        with torch.no_grad():
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores_np = scores.cpu().numpy()
            results.extend(scores_np)
    return results


def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()
