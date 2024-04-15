from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import sort_and_select_top_k


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

    rerank_scores = parallel_process_upr(queries, contents_list, prefix_prompt, suffix_prompt, tokenizer,
                                         device, model, shard_size)

    sorted_contents, sorted_ids, sorted_scores = sort_and_select_top_k(contents_list, ids_list, rerank_scores, top_k)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sorted_contents, sorted_ids, sorted_scores


def process_single_query(query: str, contents: List[str], prefix_prompt: str, suffix_prompt: str, tokenizer, device,
                         model, shard_size: int) -> List[float]:
    # Load the model in the function to ensure it's loaded in each process
    if device == 'cuda':
        model = model.to(device)

    # The rest of the function remains mostly unchanged, just ensure model is loaded and moved to the correct device
    prompts = [f"{prefix_prompt} {content} {suffix_prompt}" for content in contents]
    context_tokens = tokenizer(prompts, padding='longest', max_length=512, pad_to_multiple_of=8, truncation=True,
                               return_tensors='pt')
    context_tensor, context_attention_mask = context_tokens.input_ids, context_tokens.attention_mask
    if device == 'cuda':
        context_tensor, context_attention_mask = context_tensor.cuda(), context_attention_mask.cuda()

    question_tokens = tokenizer([query], max_length=128, truncation=True, return_tensors='pt')
    question_tensor = question_tokens.input_ids
    if device == 'cuda':
        question_tensor = question_tensor.cuda()
    question_tensor = torch.repeat_interleave(question_tensor, len(contents), dim=0)

    sharded_nll_list = []
    for i in range(0, len(context_tensor), shard_size):
        encoder_tensor_view = context_tensor[i: i + shard_size]
        attention_mask_view = context_attention_mask[i: i + shard_size]
        decoder_tensor_view = question_tensor[i: i + shard_size]
        with torch.no_grad():
            logits = model(input_ids=encoder_tensor_view, attention_mask=attention_mask_view,
                           labels=decoder_tensor_view).logits
        log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        nll = -log_softmax.gather(2, decoder_tensor_view.unsqueeze(2)).squeeze(2)
        avg_nll = torch.sum(nll, dim=1)
        sharded_nll_list.append(avg_nll)

    return list(map(lambda x: -float(x), sharded_nll_list[0]))


def parallel_process_upr(queries: List[str], contents: List[List[str]], prefix_prompt: str, suffix_prompt: str,
                         tokenizer, device, model, shard_size: int) -> List[List[float]]:
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_query, query, content, prefix_prompt, suffix_prompt, tokenizer, device,
                            model, shard_size) for query, content in zip(queries, contents)]
        results = [future.result() for future in futures]
    return results
