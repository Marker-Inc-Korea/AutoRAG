import asyncio
from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import process_batch


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

    # Run async cohere_rerank_pure function
    tasks = [get_colbert_score(query, document, model, tokenizer) for query, document, ids in
             zip(queries, contents_list, ids_list)]
    loop = asyncio.get_event_loop()
    score_results = loop.run_until_complete(process_batch(tasks, batch_size=batch))

    del model

    def rerank_results(contents, ids, scores, top_k):
        reranked_content, reranked_id, reranked_score = zip(
            *sorted(zip(contents, ids, scores), key=lambda x: x[2], reverse=True))
        return list(reranked_content)[:top_k], list(reranked_id)[:top_k], list(reranked_score)[:top_k]

    reranked_contents_list, reranked_ids_list, reranked_scores_list = zip(*list(map(
        rerank_results, contents_list, ids_list, score_results, [top_k] * len(contents_list))))
    return list(reranked_contents_list), list(reranked_ids_list), list(reranked_scores_list)


async def get_colbert_score(query: str, content_list: List[str],
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
