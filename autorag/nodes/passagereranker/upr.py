import logging
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import ray
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from autorag.nodes.passagereranker.base import passage_reranker_node
from autorag.utils.util import select_top_k, sort_by_scores

logger = logging.getLogger("AutoRAG")


@passage_reranker_node
def upr(queries: List[str], contents_list: List[List[str]],
        scores_list: List[List[float]], ids_list: List[List[str]],
        top_k: int, use_bf16: bool = False,
        prefix_prompt: str = "Passage: ",
        suffix_prompt: str = "Please write a question based on this passage.",
        num_gpus: int = 1) \
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
    :param use_bf16: Whether to use bfloat16 for the model. Default is False.
    :param prefix_prompt: The prefix prompt for the language model that generates question for reranking.
        Default is "Passage: ".
        The prefix prompt serves as the initial context or instruction for the language model.
        It sets the stage for what is expected in the output
    :param suffix_prompt: The suffix prompt for the language model that generates question for reranking.
        Default is "Please write a question based on this passage.".
        The suffix prompt provides a cue or a closing instruction to the language model,
            signaling how to conclude the generated text or what format to follow at the end.
    :param num_gpus: The number of GPUs that you want to use. Default is 1.
        Set to 1 if you use a CPU or a single GPU.
    :return: tuple of lists containing the reranked contents, ids, and scores
    """
    df = pd.DataFrame({
        'query': queries,
        'contents': contents_list,
        'ids': ids_list,
    })
    ds = ray.data.from_pandas(df)

    scorer = UPRScorer(suffix_prompt=suffix_prompt, prefix_prompt=prefix_prompt, use_bf16=use_bf16)

    if torch.cuda.is_available():
        score_batch = ds.map_batches(scorer, batch_size=1,
                                     concurrency=num_gpus, num_gpus=1)
    else:
        score_batch = ds.map_batches(scorer,
                                     batch_size=1,
                                     concurrency=min(len(df), os.cpu_count()), num_cpus=os.cpu_count())
    scores = score_batch.to_pandas()['output'].tolist()  # converted to a flatten list of scores

    del scorer

    explode_df = df.explode('contents')
    explode_df['scores'] = scores
    df['scores'] = explode_df.groupby(level=0, sort=False)['scores'].apply(list).tolist()
    df[['contents', 'ids', 'scores']] = df.apply(lambda x: sort_by_scores(x, reverse=False), axis=1,
                                                 result_type='expand')
    results = select_top_k(df, ['contents', 'ids', 'scores'], top_k)
    return results['contents'].tolist(), results['ids'].tolist(), results['scores'].tolist()


class UPRScorer:
    def __init__(self, suffix_prompt: str, prefix_prompt: str,
                 use_bf16: bool = False):
        model_name = "t5-large"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                                torch_dtype=torch.bfloat16
                                                                if use_bf16 else torch.float32).to(self.device)
        self.suffix_prompt = suffix_prompt
        self.prefix_prompt = prefix_prompt

    def __call__(self, batch: Dict[str, np.ndarray]):
        query_token = self.tokenizer(batch['query'][0], max_length=128, truncation=True, return_tensors='pt')
        contents = batch['contents'][0]
        prompts = list(map(lambda content: f"{self.prefix_prompt} {content} {self.suffix_prompt}", contents))
        prompt_token_outputs = self.tokenizer(prompts, padding='longest',
                                              max_length=512,
                                              pad_to_multiple_of=8,
                                              truncation=True,
                                              return_tensors='pt')

        query_input_ids = torch.repeat_interleave(query_token['input_ids'], len(contents),
                                                  dim=0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=prompt_token_outputs['input_ids'].to(self.device),
                                attention_mask=prompt_token_outputs['attention_mask'].to(self.device),
                                labels=query_input_ids).logits
        log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        nll = -log_softmax.gather(2, query_input_ids.unsqueeze(2)).squeeze(2)
        avg_nll = torch.sum(nll, dim=1)
        return {"output": avg_nll.tolist()}

    def __del__(self):
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
