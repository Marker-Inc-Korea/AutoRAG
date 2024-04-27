from typing import List, Tuple

import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import select_top_k, flatten_apply, sort_by_scores


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                       torch_dtype=torch.bfloat16
                                                       if use_bf16 else torch.float32).to(device)

    prompt_tokens = flatten_apply(make_content_prompt, contents_list,
                                  prefix_prompt=prefix_prompt, suffix_prompt=suffix_prompt,
                                  tokenizer=tokenizer)

    df = pd.DataFrame({
        'query': queries,
        'contents': contents_list,
        'ids': ids_list,
        'prompt_tokens': prompt_tokens,
    })
    df['scores'] = df.apply(lambda x: get_upr_score(x['prompt_tokens'], x['query'], model, tokenizer, device), axis=1)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df[['contents', 'ids', 'scores']] = df.apply(lambda x: sort_by_scores(x, reverse=False), axis=1,
                                                 result_type='expand')
    results = select_top_k(df, ['contents', 'ids', 'scores'], top_k)
    return results['contents'].tolist(), results['ids'].tolist(), results['scores'].tolist()


def make_content_prompt(contents: List[str], prefix_prompt: str, suffix_prompt: str,
                        tokenizer):
    prompts = list(map(lambda content: f"{prefix_prompt} {content} {suffix_prompt}", contents))
    prompt_token_outputs = tokenizer(prompts, padding='longest',
                                     max_length=512,
                                     pad_to_multiple_of=8,
                                     truncation=True,
                                     return_tensors='pt')
    input_id_list = list(prompt_token_outputs['input_ids'].chunk(len(contents), dim=0))
    attn_mask_list = list(prompt_token_outputs['attention_mask'].chunk(len(contents), dim=0))
    return list(map(lambda x, y: {'input_ids': x, 'attention_mask': y}, input_id_list, attn_mask_list))


def get_upr_score(content_tokens: List, query: str,
                  model, tokenizer, device) -> List[float]:
    query_token = tokenizer(query, max_length=128, truncation=True, return_tensors='pt')
    query_input_ids = torch.repeat_interleave(query_token['input_ids'], len(content_tokens), dim=0).to(device)

    # concat list of content tensors
    content_input_ids = torch.cat(list(map(lambda x: x['input_ids'], content_tokens)), dim=0).to(device)
    content_attn_mask = torch.cat(list(map(lambda x: x['attention_mask'], content_tokens)), dim=0).to(device)

    logits = model(input_ids=content_input_ids, attention_mask=content_attn_mask,
                   labels=query_input_ids).logits
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    nll = -log_softmax.gather(2, query_input_ids.unsqueeze(2)).squeeze(2)
    avg_nll = torch.sum(nll, dim=1)
    return avg_nll.tolist()
