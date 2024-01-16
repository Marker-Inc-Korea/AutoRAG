from typing import List, Tuple
from uuid import UUID, uuid4
from transformers import AutoModel, AutoTokenizer

import asyncio
import torch


def UPR_rerank(queries: List[str], contents_list: List[List[str]],
               scores_list: List[List[float]], ids_list: List[List[UUID]], shard_size: int = 16, prefix_prompt: str = "",
               suffix_prompt: str = "",
               model_name: str = "t5-large", device: str = "gpu", torch_dtype=torch.float16) -> List[Tuple[List[str]]]:
    """
    UPRReranker is a reranker based on UPR (https://github.com/DevSinghSachan/unsupervised-passage-reranking).
    The language model will make a question based on the passage and rerank the passages by the likelihood of the question.
    """
    model = AutoModel.from_pretrained(model_name, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tasks = [UPR_rerank_pure(query, contents, scores, ids, model, tokenizer, device, shard_size, prefix_prompt, suffix_prompt) for query, contents, scores, ids in
             zip(queries, contents_list, scores_list, ids_list)]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    return results


async def UPR_rerank_pure(query: str, contents: List[str], scores: List[float], ids: List[UUID],
                          model: AutoModel = AutoModel.from_pretrained("t5-large", torch_dtype=torch.float16),
                          tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("t5-large"), device: str = "gpu",
                          shard_size: int = 16, prefix_prompt: str = "Passage: ",
                          suffix_prompt: str = "Please write a question based on this passage.") -> Tuple[List[str]]:
    """
    Rerank a list of contents based on their relevance to a query using UPR.
    :param query: str: The query that UPR rerank contents according to
    :param contents: List[str]: The contents that is going to
    :param scores: List[float]: scores of contents from last node(Retrieval or Reranker).
    :param ids: List[UUID]: IDs of contents
    :param model: transformers.Automodel: LLM model to score contents for rerank
    :param device: string: choose "gpu" if not, this version UPR
    :param tokenizer: transformers.AutoTokenizer
    :return: tuple of lists containing the reranked contents, ids, and scores
    """

    indexes, scores = calculate_likelihood(model=model, tokenizer=tokenizer, query=query, contents=contents,
                                           shard_size=shard_size, device=device, prefix_prompt=prefix_prompt,
                                           suffix_prompt=suffix_prompt)
    reranked_contents, scores, ids = zip(*[list(zip(contents, scores, ids))[idx] for idx in indexes])

    return tuple(reranked_contents, scores, ids)


def calculate_likelihood(model: AutoModel, tokenizer: AutoTokenizer, query: str, contents: List[str], shard_size: int,
                         device: str = "gpu", prefix_prompt: str = "", suffix_prompt: str = "") -> tuple[
    List[int], List[float]]:
    prompts = [f"{prefix_prompt}{content}{suffix_prompt}" for content in contents]
    # tokenize contents and instruction prompts
    content_tokens = tokenizer(prompts,
                               padding='longest',
                               max_length=512,
                               pad_to_multiple_of=8,
                               truncation=True,
                               return_tensors='pt')
    content_tensor, content_attention_mask = content_tokens.input_ids, content_tokens.attention_mask
    if device in ["gpu", "GPU"]:
        content_tensor, content_attention_mask = content_tensor.cuda(), content_attention_mask.cuda()

    # tokenize question
    query_tokens = tokenizer([query],
                             max_length=128,
                             truncation=True,
                             return_tensors='pt')
    query_tensor = query_tokens.input_ids
    if device in ["gpu", "GPU"]:
        query_tensor = query_tensor.cuda()
    query_tensor = torch.repeat_interleave(query_tensor, len(contents), dim=0)

    sharded_nll_list = []

    # calculate log likelihood
    for i in range(0, len(content_tensor), shard_size):
        encoder_tensor_view = content_tensor[i: i + shard_size]
        attention_mask_view = content_attention_mask[i: i + shard_size]
        decoder_tensor_view = query_tensor[i: i + shard_size]
        with torch.no_grad():
            logits = model(input_ids=encoder_tensor_view,
                           attention_mask=attention_mask_view,
                           labels=decoder_tensor_view).logits

        log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        nll = -log_softmax.gather(2, decoder_tensor_view.unsqueeze(2)).squeeze(2)

        avg_nll = torch.sum(nll, dim=1)
        sharded_nll_list.append(avg_nll)

    topk_scores, indexes = torch.topk(-torch.cat(sharded_nll_list), k=len(content_tensor))

    return indexes.tolist(), topk_scores.tolist()