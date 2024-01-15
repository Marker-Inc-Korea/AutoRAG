from typing import List, Tuple
from uuid import UUID

import torch
import asyncio

from transformers import T5Tokenizer, T5ForConditionalGeneration


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


def monot5(queries: List[str], contents_list: List[List[str]],
           scores_list: List[List[float]], ids_list: List[List[UUID]],
           model_name: str = 'castorini/monot5-3b-msmarco-10k') -> List[Tuple[List[str]]]:
    # Load the tokenizer and model from the pre-trained MonoT5 model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).eval()
    # Retrieve the tokens used by the model to represent false and true predictions
    token_false, token_true = prediction_tokens[model_name]
    token_false_id = tokenizer.convert_tokens_to_ids(token_false)
    token_true_id = tokenizer.convert_tokens_to_ids(token_true)
    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run async mono_t5_rerank_pure function
    tasks = [mono_t5_pure(query, contents, scores, ids, model, device, tokenizer, token_false_id, token_true_id) \
             for query, contents, scores, ids in zip(queries, contents_list, scores_list, ids_list)]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    return results


async def mono_t5_pure(query: str, contents: List[str], scores: List[float], ids: List[UUID],
                       model, device, tokenizer, token_false_id, token_true_id) -> Tuple[List[str]]:
    """
    Rerank a list of contents based on their relevance to a query using MonoT5.
    :param query: The query to use for reranking
    :param contents: The list of contents to rerank
    :param scores: The list of scores retrieved from the initial ranking
    :param ids: The list of ids retrieved from the initial ranking
    :param model: The MonoT5 model to use for reranking
    :param device: The device to run the model on (GPU if available, otherwise CPU)
    :param tokenizer: The tokenizer to use for the model
    :param token_false_id: The id of the token used by the model to represent a false prediction
    :param token_true_id: The id of the token used by the model to represent a true prediction
    :return: tuple of lists containing the reranked contents, ids, and scores
    """
    model.to(device)

    # Format the input for the model by combining each content with the query
    input_texts = [f'Query: {query} Document: {content}' for content in contents]
    # Tokenize the input texts and prepare for model input
    input_encodings = tokenizer(input_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(
        device)

    # Generate model predictions without updating model weights
    with torch.no_grad():
        outputs = model.generate(input_ids=input_encodings['input_ids'],
                                 attention_mask=input_encodings['attention_mask'],
                                 output_scores=True,
                                 return_dict_in_generate=True)

    # Extract logits for the 'false' and 'true' tokens from the model's output
    logits = outputs.scores[-1][:, [token_false_id, token_true_id]]
    # Calculate the softmax probability of the 'true' token
    probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1]  # Get the probability of the 'true' token

    # Create a list of tuples pairing each content with its relevance probability
    content_ids_probs = list(zip(contents, ids, probs.tolist()))

    # Sort the list of pairs based on the relevance score in descending order
    sorted_content_ids_probs = sorted(content_ids_probs, key=lambda x: x[2], reverse=True)

    return tuple(map(list, zip(*sorted_content_ids_probs)))
