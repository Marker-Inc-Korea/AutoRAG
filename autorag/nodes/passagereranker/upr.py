from typing import List, Tuple

import torch
from click.core import F
from transformers import T5Tokenizer, T5ForConditionalGeneration

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import select_top_k, flatten_apply, make_batch


@passage_reranker_node
def upr(queries: List[str], contents_list: List[List[str]],
        scores_list: List[List[float]], ids_list: List[List[str]],
        top_k: int, shard_size: int = 16, use_bf16: bool = False,
        prefix_prompt: str = "Passage: ",
        suffix_prompt: str = "Please write a question based on this passage.",
        batch: int = 64) \
        -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Rerank a list of contents based on their relevance to a query using UPR.
    UPR is a reranker based on UPR (https://github.com/DevSinghSachan/unsupervised-passage-reranking).
    The language model will make a question based on the passage and rerank the passages by the likelihood of the question.
    The default model is t5-large.

    :param queries: The list of queries to use for reranking
    :param contents_list: The list of lists of contents to rerank
    :param scores_list: The list of lists of scores retrieved from the initial ranking
    :param ids_list: The list of lists of ids retrieved from the initial ranking
    :param top_k: The number of passages to be retrieved
    :param shard_size: The shard size for the model.
        The larger the shard size, the faster the reranking speed.
        But it will consume more memory and compute power.
        Default is 16.
    :param use_bf16: Whether to use bfloat16 for the model. Default is False.
    :param prefix_prompt: The prefix prompt for the language model that generates question for reranking.
        Default is "Passage: ".
        The prefix prompt serves as the initial context or instruction for the language model.
        It sets the stage for what is expected in the output
    :param suffix_prompt: The suffix prompt for the language model that generates question for reranking.
        Default is "Please write a question based on this passage.".
        The suffix prompt provides a cue or a closing instruction to the language model,
            signaling how to conclude the generated text or what format to follow at the end.
    :param batch: The number of queries to be processed in a batch
    :return: tuple of lists containing the reranked contents, ids, and scores
    """
    # Load the tokenizer and model
    model_name = "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                       torch_dtype=torch.bfloat16 if use_bf16 else torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nested_list = [list(map(lambda x: [query, f"{prefix_prompt} {x} {suffix_prompt}"], content_list))
                   for query, content_list in zip(queries, contents_list)]

    rerank_scores = flatten_apply(upr_run_model, nested_list, model=model, tokenizer=tokenizer, batch_size=batch,
                                  device=device, shard_size=shard_size)

    sorted_contents, sorted_ids, sorted_scores = select_top_k(contents_list, ids_list, rerank_scores, top_k)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sorted_contents, sorted_ids, sorted_scores


def upr_run_model(input_texts, model, tokenizer, device, batch_size: int, shard_size: int):
    batch_input_texts = make_batch(input_texts, batch_size)
    results = []

    for batch_texts in batch_input_texts:
        flattened_batch_queries, flattened_batch_prompts = map(list, zip(*batch_texts))

        # tokenize contexts and instruction prompts
        context_tokens = tokenizer(flattened_batch_prompts,
                                   padding='longest',
                                   max_length=512,
                                   pad_to_multiple_of=8,
                                   truncation=True,
                                   return_tensors='pt').to(device)
        context_tensor, context_attention_mask = context_tokens.input_ids, context_tokens.attention_mask
        if device == 'cuda':
            context_tensor, context_attention_mask = context_tensor.cuda(), context_attention_mask.cuda()

        question_tokens = tokenizer(flattened_batch_queries,
                                    padding='longest',
                                    max_length=512,
                                    pad_to_multiple_of=8,
                                    truncation=True,
                                    return_tensors='pt').to(device)
        question_tensor = question_tokens.input_ids
        if device == 'cuda':
            question_tensor = question_tensor.cuda()

        # calculate log likelihood
        for i in range(0, len(context_tensor), shard_size):
            encoder_tensor_view = context_tensor[i: i + shard_size]
            attention_mask_view = context_attention_mask[i: i + shard_size]
            decoder_tensor_view = question_tensor[i: i + shard_size]
            with torch.no_grad():
                logits = model(input_ids=encoder_tensor_view,
                               attention_mask=attention_mask_view,
                               labels=decoder_tensor_view).logits
            normalized_scores = [float(score[1]) for score in F.softmax(logits, dim=1)]
            results.extend(normalized_scores)
    return results
