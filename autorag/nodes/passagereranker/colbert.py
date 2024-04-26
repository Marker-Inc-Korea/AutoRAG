from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import flatten_apply


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

    # get query and content embeddings
    query_embedding_tensor = get_colbert_embedding_batch(queries, model, tokenizer, batch)
    query_embedding = torch.cat(query_embedding_tensor, dim=0)
    content_embedding_list = flatten_apply(get_colbert_embedding_batch, contents_list, model=model, tokenizer=tokenizer,
                                           batch_size=batch)

    del model

    def rerank_results(contents, ids, scores, top_k):
        reranked_content, reranked_id, reranked_score = zip(
            *sorted(zip(contents, ids, scores), key=lambda x: x[2], reverse=True))
        return list(reranked_content)[:top_k], list(reranked_id)[:top_k], list(reranked_score)[:top_k]

    reranked_contents_list, reranked_ids_list, reranked_scores_list = zip(*list(map(
        rerank_results, contents_list, ids_list, score_results, [top_k] * len(contents_list))))
    return list(reranked_contents_list), list(reranked_ids_list), list(reranked_scores_list)


def get_colbert_embedding_batch(input_strings: List[str],
                                model, tokenizer, batch_size: int) -> List[torch.Tensor]:
    encoding = tokenizer(input_strings, return_tensors="pt", padding=True)
    input_batches = slice_tokenizer_result(encoding, batch_size)
    result_embedding = []
    for encoding in input_batches:
        result_embedding.append(model(**encoding).last_hidden_state)
    total_tensor = torch.cat(result_embedding, dim=0)  # shape [batch_size, token_length, embedding_dim]
    return list(total_tensor.chunk(total_tensor.size()[0]))


def slice_tokenizer_result(tokenizer_output, batch_size):
    input_ids_batches = slice_tensor(tokenizer_output["input_ids"], batch_size)
    attention_mask_batches = slice_tensor(tokenizer_output["attention_mask"], batch_size)
    token_type_ids_batches = slice_tensor(tokenizer_output.get("token_type_ids", None), batch_size)
    return [{"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
            for input_ids, attention_mask, token_type_ids in
            zip(input_ids_batches, attention_mask_batches, token_type_ids_batches)]


def slice_tensor(input_tensor, batch_size):
    # Calculate the number of full batches
    num_full_batches = input_tensor.size(0) // batch_size

    # Slice the tensor into batches
    tensor_list = [input_tensor[i * batch_size:(i + 1) * batch_size] for i in range(num_full_batches)]

    # Handle the last batch if it's smaller than batch_size
    remainder = input_tensor.size(0) % batch_size
    if remainder:
        tensor_list.append(input_tensor[-remainder:])

    return tensor_list


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
