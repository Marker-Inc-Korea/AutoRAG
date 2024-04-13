from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import make_batch, sort_and_select_top_k


@passage_reranker_node
def colbert_reranker(queries: List[str], contents_list: List[List[str]],
                     scores_list: List[List[float]], ids_list: List[List[str]],
                     top_k: int, batch: int = 64,
                     model_name: str = "colbert-ir/colbertv2.0",
                     ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Rerank a list of contents with Colbert rerank models.
    You can get more information about a Colbert model at https://huggingface.co/colbert-ir/colbertv2.0.
    It uses BERT-based model, so recommend using CUDA gpu for faster reranking.

    :param queries: The list of queries to use for reranking
    :param contents_list: The list of lists of contents to rerank
    :param scores_list: The list of lists of scores retrieved from the initial ranking
    :param ids_list: The list of lists of ids retrieved from the initial ranking
    :param top_k: The number of passages to be retrieved
    :param batch: The number of queries to be processed in a batch
        Default is 64.
    :param model_name: The model name for Colbert rerank.
        You can choose colbert model for reranking.
        Default is "colbert-ir/colbertv2.0".
    :return: Tuple of lists containing the reranked contents, ids, and scores
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rerank_scores = colbert_run_model(queries, contents_list, model, tokenizer, batch_size=batch)

    sorted_contents, sorted_ids, sorted_scores = sort_and_select_top_k(contents_list, ids_list, rerank_scores, top_k)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sorted_contents, sorted_ids, sorted_scores


def colbert_run_model(queries, contents_list, model, tokenizer, batch_size: int):
    batch_queries_list = make_batch(queries, batch_size)
    batch_contents_list = make_batch(contents_list, batch_size)

    results = list(map(lambda pair: get_colbert_score(*pair, model, tokenizer),
                       zip(sum(batch_queries_list, []), sum(batch_contents_list, []))))

    return results


def get_colbert_score(query: str, content_list: List[str],
                      model, tokenizer) -> List[float]:
    query_encoding = tokenizer(query, return_tensors="pt")
    query_embedding = model(**query_encoding).last_hidden_state
    rerank_score_list = []

    for document_text in content_list:
        document_encoding = tokenizer(
            document_text, return_tensors="pt", truncation=True, max_length=512
        )
        document_embedding = model(**document_encoding).last_hidden_state

        sim_matrix = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(2), document_embedding.unsqueeze(1), dim=-1
        )

        # Take the maximum similarity for each query token (across all document tokens)
        # sim_matrix shape: [batch_size, query_length, doc_length]
        max_sim_scores, _ = torch.max(sim_matrix, dim=2)
        rerank_score_list.append(torch.mean(max_sim_scores, dim=1))

    return list(map(float, rerank_score_list))
