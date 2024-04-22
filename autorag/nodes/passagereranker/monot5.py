from itertools import chain
from typing import List, Tuple

import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import make_batch, sort_by_scores, flatten_apply, select_top_k

prediction_tokens = {
    'castorini/monot5-base-msmarco': ['▁false', '▁true'],
    'castorini/monot5-base-msmarco-10k': ['▁false', '▁true'],
    'castorini/monot5-large-msmarco': ['▁false', '▁true'],
    'castorini/monot5-large-msmarco-10k': ['▁false', '▁true'],
    'castorini/monot5-base-med-msmarco': ['▁false', '▁true'],
    'castorini/monot5-3b-med-msmarco': ['▁false', '▁true'],
    'castorini/monot5-3b-msmarco-10k': ['▁false', '▁true'],
    'unicamp-dl/mt5-base-en-msmarco': ['▁no', '▁yes'],
    'unicamp-dl/ptt5-base-pt-msmarco-10k-v2': ['▁não', '▁sim'],
    'unicamp-dl/ptt5-base-pt-msmarco-100k-v2': ['▁não', '▁sim'],
    'unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2': ['▁não', '▁sim'],
    'unicamp-dl/mt5-base-en-pt-msmarco-v2': ['▁no', '▁yes'],
    'unicamp-dl/mt5-base-mmarco-v2': ['▁no', '▁yes'],
    'unicamp-dl/mt5-base-en-pt-msmarco-v1': ['▁no', '▁yes'],
    'unicamp-dl/mt5-base-mmarco-v1': ['▁no', '▁yes'],
    'unicamp-dl/ptt5-base-pt-msmarco-10k-v1': ['▁não', '▁sim'],
    'unicamp-dl/ptt5-base-pt-msmarco-100k-v1': ['▁não', '▁sim'],
    'unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1': ['▁não', '▁sim'],
    'unicamp-dl/mt5-3B-mmarco-en-pt': ['▁', '▁true'],
    'unicamp-dl/mt5-13b-mmarco-100k': ['▁', '▁true'],
}


@passage_reranker_node
def monot5(queries: List[str], contents_list: List[List[str]],
           scores_list: List[List[float]], ids_list: List[List[str]],
           top_k: int, model_name: str = 'castorini/monot5-3b-msmarco-10k',
           batch: int = 64, ) \
        -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Rerank a list of contents based on their relevance to a query using MonoT5.

    :param queries: The list of queries to use for reranking
    :param contents_list: The list of lists of contents to rerank
    :param scores_list: The list of lists of scores retrieved from the initial ranking
    :param ids_list: The list of lists of ids retrieved from the initial ranking
    :param top_k: The number of passages to be retrieved
    :param model_name: The name of the MonoT5 model to use for reranking
        Note: default model name is 'castorini/monot5-3b-msmarco-10k'
            If there is a '/' in the model name parameter,
            when we create the file to store the results, the path will be twisted because of the '/'.
            Therefore, it will be received as '_' instead of '/'.
    :param batch: The number of queries to be processed in a batch
    :return: tuple of lists containing the reranked contents, ids, and scores
    """
    # replace '_' to '/'
    if '_' in model_name:
        model_name = model_name.replace('_', '/')
    # Load the tokenizer and model from the pre-trained MonoT5 model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).eval()
    # Retrieve the tokens used by the model to represent false and true predictions
    token_false, token_true = prediction_tokens[model_name]
    token_false_id = tokenizer.convert_tokens_to_ids(token_false)
    token_true_id = tokenizer.convert_tokens_to_ids(token_true)
    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    nested_list = [list(map(lambda x: [f'Query: {query} Document: {x}'], content_list))
                   for query, content_list in zip(queries, contents_list)]

    rerank_scores = flatten_apply(monot5_run_model, nested_list, model=model, batch_size=batch, tokenizer=tokenizer,
                                  device=device, token_false_id=token_false_id, token_true_id=token_true_id)

    df = pd.DataFrame({
        'contents': contents_list,
        'ids': ids_list,
        'scores': rerank_scores,
    })
    df[['contents', 'ids', 'scores']] = df.apply(sort_by_scores, axis=1, result_type='expand')
    results = select_top_k(df, ['contents', 'ids', 'scores'], top_k)

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results['contents'].tolist(), results['ids'].tolist(), results['scores'].tolist()


def monot5_run_model(input_texts, model, batch_size: int, tokenizer, device, token_false_id, token_true_id):
    batch_input_texts = make_batch(input_texts, batch_size)
    results = []
    for batch_texts in batch_input_texts:
        flattened_batch_texts = list(chain.from_iterable(batch_texts))
        input_encodings = tokenizer(flattened_batch_texts, padding=True, truncation=True, max_length=512,
                                    return_tensors='pt').to(
            device)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_encodings['input_ids'],
                                     attention_mask=input_encodings['attention_mask'],
                                     output_scores=True,
                                     return_dict_in_generate=True)

        # Extract logits for the 'false' and 'true' tokens from the model's output
        logits = outputs.scores[-1][:, [token_false_id, token_true_id]]
        # Calculate the softmax probability of the 'true' token
        probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
        results.extend(probs.tolist())
    return results
