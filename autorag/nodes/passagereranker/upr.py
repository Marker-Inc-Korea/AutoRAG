from typing import List, Tuple

from transformers import T5Tokenizer, T5ForConditionalGeneration

import asyncio
import torch


def upr(queries: List[str], contents_list: List[List[str]],
        scores_list: List[List[float]], ids_list: List[List[str]],
        top_k: int, shard_size: int = 16, model_name: str = "t5-large",
        use_bf16: bool = False, prefix_prompt: str = "Passage: ",
        suffix_prompt: str = "Please write a question based on this passage.") \
        -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    UPRReranker is a reranker based on UPR (https://github.com/DevSinghSachan/unsupervised-passage-reranking).
    The language model will make a question based on the passage and rerank the passages by the likelihood of the question.
    """
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                       torch_dtype=torch.bfloat16 if use_bf16 else torch.float32)
    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run async upr_rerank_pure function
    tasks = [upr_pure(query, contents, scores, ids, top_k, model, tokenizer, device, shard_size, prefix_prompt,
                      suffix_prompt) for query, contents, scores, ids in
             zip(queries, contents_list, scores_list, ids_list)]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    content_result = list(map(lambda x: x[0], results))
    id_result = list(map(lambda x: x[1], results))
    score_result = list(map(lambda x: x[2], results))
    return content_result, id_result, score_result


async def upr_pure(query: str, contents: List[str], scores: List[float],
                   ids: List[str], top_k: int, model, device, tokenizer,
                   shard_size: int, prefix_prompt: str, suffix_prompt: str) \
        -> Tuple[List[str], List[str], List[float]]:
    indexes, scores = calculate_likelihood(model=model, tokenizer=tokenizer, query=query, contents=contents,
                                           shard_size=shard_size, device=device, prefix_prompt=prefix_prompt,
                                           suffix_prompt=suffix_prompt)
    reranked_contents, reranked_ids = zip(*[(contents[idx], ids[idx]) for idx in indexes])

    # crop with top_k
    if len(reranked_contents) < top_k:
        top_k = len(reranked_contents)
    reranked_contents = reranked_contents[:top_k]
    reranked_ids = reranked_ids[:top_k]
    scores = scores[:top_k]

    return reranked_contents, reranked_ids, scores


def calculate_likelihood(question: str, contents: List[str],
                         prefix_prompt: str, suffix_prompt: str,
                         tokenizer, device, model, shard_size: int, ) -> tuple[List[int], List[float]]:
    # create prompts
    prompts = [f"{prefix_prompt} {content} {suffix_prompt}" for content in contents]

    # tokenize contexts and instruction prompts
    context_tokens = tokenizer(prompts,
                               padding='longest',
                               max_length=512,
                               pad_to_multiple_of=8,
                               truncation=True,
                               return_tensors='pt')
    context_tensor, context_attention_mask = context_tokens.input_ids, context_tokens.attention_mask
    if device.type == 'cuda':
        context_tensor, context_attention_mask = context_tensor.cuda(), context_attention_mask.cuda()

    # tokenize question
    question_tokens = tokenizer([question],
                                max_length=128,
                                truncation=True,
                                return_tensors='pt')
    question_tensor = question_tokens.input_ids
    if device.type == 'cuda':
        question_tensor = question_tensor.cuda()
    question_tensor = torch.repeat_interleave(question_tensor, len(contents), dim=0)

    sharded_nll_list = []

    # calculate log likelihood
    for i in range(0, len(context_tensor), shard_size):
        encoder_tensor_view = context_tensor[i: i + shard_size]
        attention_mask_view = context_attention_mask[i: i + shard_size]
        decoder_tensor_view = question_tensor[i: i + shard_size]
        with torch.no_grad():
            logits = model(input_ids=encoder_tensor_view,
                           attention_mask=attention_mask_view,
                           labels=decoder_tensor_view).logits

        log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        nll = -log_softmax.gather(2, decoder_tensor_view.unsqueeze(2)).squeeze(2)

        avg_nll = torch.sum(nll, dim=1)
        sharded_nll_list.append(avg_nll)

    topk_scores, indexes = torch.topk(-torch.cat(sharded_nll_list), k=len(context_tensor))

    return indexes.tolist(), topk_scores.tolist()
