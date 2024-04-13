from typing import List, Tuple

import torch
import torch.nn.functional as F

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.nodes.passagereranker.tart.modeling_enc_t5 import EncT5ForSequenceClassification
from autorag.nodes.passagereranker.tart.tokenization_enc_t5 import EncT5Tokenizer
from autorag.utils.util import make_batch, sort_and_select_top_k


@passage_reranker_node
def tart(queries: List[str], contents_list: List[List[str]],
         scores_list: List[List[float]], ids_list: List[List[str]],
         top_k: int, instruction: str = "Find passage to answer given question",
         batch: int = 64) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Rerank a list of contents based on their relevance to a query using Tart.
    TART is a reranker based on TART (https://github.com/facebookresearch/tart).
    You can rerank the passages with the instruction using TARTReranker.
    The default model is facebook/tart-full-flan-t5-xl.

    :param queries: The list of queries to use for reranking
    :param contents_list: The list of lists of contents to rerank
    :param scores_list: The list of lists of scores retrieved from the initial ranking
    :param ids_list: The list of lists of ids retrieved from the initial ranking
    :param top_k: The number of passages to be retrieved
    :param instruction: The instruction for reranking.
        Note: default instruction is "Find passage to answer given question"
            The default instruction from the TART paper is being used.
            If you want to use a different instruction, you can change the instruction through this parameter
    :param batch: The number of queries to be processed in a batch
    :return: tuple of lists containing the reranked contents, ids, and scores
    """
    model_name = "facebook/tart-full-flan-t5-xl"
    model = EncT5ForSequenceClassification.from_pretrained(model_name)
    tokenizer = EncT5Tokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    features = []
    for query, contents in zip(queries, contents_list):
        instruction_query = ['{} [SEP] {}'.format(instruction, query)] * len(contents)
        feature = tokenizer(instruction_query, contents, padding=True, truncation=True, return_tensors="pt").to(device)
        features.append(feature)

    rerank_scores = tart_run_model(features, model=model, batch_size=batch)

    sorted_contents, sorted_ids, sorted_scores = sort_and_select_top_k(contents_list, ids_list, rerank_scores, top_k)

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sorted_contents, sorted_ids, sorted_scores


def tart_run_model(features, model, batch_size: int):
    batch_features = make_batch(features, batch_size)
    results = []
    for batch_feature in batch_features:
        for feature in batch_feature:
            with torch.no_grad():
                pred_scores = model(**feature).logits
                normalized_scores = [float(score[1]) for score in F.softmax(pred_scores, dim=1)]
            results.append(normalized_scores)
    return results
