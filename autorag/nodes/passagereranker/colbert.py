from typing import List, Tuple

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import make_batch, select_top_k, flatten_apply, sort_by_scores


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
        The batch size must always be greater than 1.
    :param model_name: The model name for Colbert rerank.
        You can choose colbert model for reranking.
        Default is "colbert-ir/colbertv2.0".
    :return: Tuple of lists containing the reranked contents, ids, and scores
    """
    assert batch > 1, "Batch size must be greater than 1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    nested_list = [list(map(lambda x: [query, x], content_list)) for query, content_list in zip(queries, contents_list)]

    rerank_scores = flatten_apply(colbert_run_model, nested_list, model=model,
                                  tokenizer=tokenizer, batch_size=batch, device=device)
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


def colbert_run_model(contents_list, model, tokenizer, device, batch_size: int):
    flattened_queries, flattened_contents = map(list, zip(*contents_list))
    batch_queries = make_batch(flattened_queries, batch_size)
    batch_contents = make_batch(flattened_contents, batch_size)
    results = []

    for batch_queries, batch_contents in zip(batch_queries, batch_contents):
        # Tokenize both queries and contents together
        feature = tokenizer(batch_queries, batch_contents, padding=True, truncation=True,
                            return_tensors="pt").to(device)

        # Process the combined feature through the model in one go
        outputs = model(**feature)
        last_hidden_state = outputs.last_hidden_state

        # Split the embeddings back into query and content parts
        num_queries = len(batch_queries)
        query_embedding, content_embedding = last_hidden_state.split(
            [num_queries, last_hidden_state.size(1) - num_queries], dim=1)

        # Calculate cosine similarity between query and content embeddings
        sim_matrix = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(2), content_embedding.unsqueeze(1), dim=-1
        )

        # Take the maximum similarity for each query token (across all document tokens)
        max_sim_scores, _ = torch.max(sim_matrix, dim=2)
        results.extend(torch.mean(max_sim_scores, dim=1))

    return list(map(float, results))
