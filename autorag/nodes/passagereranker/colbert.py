from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import flatten_apply, sort_by_scores, select_top_k


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
    query_embedding_list = get_colbert_embedding_batch(queries, model, tokenizer, batch)
    content_embedding_list = flatten_apply(get_colbert_embedding_batch, contents_list, model=model, tokenizer=tokenizer,
                                           batch_size=batch)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame({
        'ids': ids_list,
        'query_embedding': query_embedding_list,
        'contents': contents_list,
        'content_embedding': content_embedding_list,
    })
    temp_df = df.explode('content_embedding')
    temp_df['score'] = temp_df.apply(lambda x: get_colbert_score(x['query_embedding'], x['content_embedding']), axis=1)
    df['scores'] = temp_df.groupby(level=0, sort=False)['score'].apply(list).tolist()
    df[['contents', 'ids', 'scores']] = df.apply(sort_by_scores, axis=1, result_type='expand')
    results = select_top_k(df, ['contents', 'ids', 'scores'], top_k)

    return results['contents'].tolist(), results['ids'].tolist(), results['scores'].tolist()


def get_colbert_embedding_batch(input_strings: List[str],
                                model, tokenizer, batch_size: int) -> List[np.array]:

    encoding = tokenizer(input_strings, return_tensors="pt", padding=True, truncation=True,
                         max_length=model.config.max_position_embeddings)

    input_batches = slice_tokenizer_result(encoding, batch_size)
    result_embedding = []
    for encoding in input_batches:
        result_embedding.append(model(**encoding).last_hidden_state)
    total_tensor = torch.cat(result_embedding, dim=0)  # shape [batch_size, token_length, embedding_dim]
    tensor_results = list(total_tensor.chunk(total_tensor.size()[0]))

    if torch.cuda.is_available():
        return list(map(lambda x: x.detach().cpu().numpy(), tensor_results))
    else:
        return list(map(lambda x: x.detach().numpy(), tensor_results))


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor_list = list(map(lambda x: x.to(device), tensor_list))

    return tensor_list


def get_colbert_score(query_embedding: np.array, content_embedding: np.array) -> float:
    query_tensor = torch.tensor(query_embedding)
    content_tensor = torch.tensor(content_embedding)
    sim_matrix = torch.nn.functional.cosine_similarity(
        query_tensor.unsqueeze(2), content_tensor.unsqueeze(1), dim=-1
    )
    max_sim_scores, _ = torch.max(sim_matrix, dim=2)
    return float(torch.mean(max_sim_scores, dim=1))
