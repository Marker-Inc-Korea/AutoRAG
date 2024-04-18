import asyncio
import functools
import itertools
import os
from typing import List, Optional

import evaluate
import pandas as pd
import sacrebleu
import torch
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import AsyncOpenAI
from rouge_score import tokenizers
from rouge_score.rouge_scorer import RougeScorer

from autorag import embedding_models
from autorag.evaluate.metric.util import calculate_cosine_similarity
from autorag.utils.util import process_batch, openai_truncate_by_token


def generation_metric(func):
    @functools.wraps(func)
    def wrapper(generation_gt: List[List[str]], generations: List[str], **kwargs) -> List[float]:
        """
        Compute generation metric.

        :param generation_gt: A list of ground truth.
            Must be 2-d list of string.
            Because it can be a multiple ground truth.
        :param generations: A list of generations that LLM generated.
        :param kwargs: The additional arguments for metric function.
        :return: A list of computed metric scores.
        """
        # make generation_gt and generations to pd dataframe
        result = list(map(lambda x: func(x[0], x[1], **kwargs), zip(generation_gt, generations)))
        return result

    return wrapper


def huggingface_evaluate(instance, key: str,
                         generation_gt: List[List[str]], generations: List[str],
                         **kwargs) -> List[float]:
    """
    Compute huggingface evaluate metric.

    :param instance: The instance of huggingface evaluates metric.
    :param key: The key to retrieve result score from huggingface evaluate result.
    :param generation_gt: A list of ground truth.
        Must be 2-d list of string.
    :param generations: A list of generations that LLM generated.
    :param kwargs: The additional arguments for metric function.
    :return: The list of scores.
    """

    def compute_score(gt: List[str], pred: str) -> float:
        return max(list(map(
            lambda x: instance.compute(predictions=[pred], references=[x], **kwargs)[key], gt)))

    result = list(map(lambda x: compute_score(x[0], x[1]), zip(generation_gt, generations)))
    return result


@generation_metric
def bleu(gt: List[str], pred: str, **kwargs) -> float:
    """
    Compute bleu score for generation.
    """
    return sacrebleu.sentence_bleu(pred, gt, **kwargs).score


def meteor(generation_gt: List[List[str]], generations: List[str],
           alpha: float = 0.9,
           beta: float = 3.0,
           gamma: float = 0.5) -> List[float]:
    """
    Compute meteor score for generation.

    :param generation_gt: A list of ground truth.
            Must be 2-d list of string.
            Because it can be a multiple ground truth.
    :param generations: A list of generations that LLM generated.
    :param alpha: Parameter for controlling relative weights of precision and recall.
        Default is 0.9.
    :param beta: Parameter for controlling shape of penalty as a
        function of as a function of fragmentation.
        Default is 3.0.
    :param gamma: Relative weight assigned to fragmentation penalty.
        Default is 0.5.
    :return: A list of computed metric scores.
    """
    meteor_instance = evaluate.load("meteor")
    result = huggingface_evaluate(meteor_instance, 'meteor', generation_gt, generations,
                                  alpha=alpha, beta=beta, gamma=gamma)
    del meteor_instance
    return result


def rouge(generation_gt: List[List[str]], generations: List[str],
          rouge_type: Optional[str] = 'rougeL',
          use_stemmer: bool = False,
          split_summaries: bool = False,
          batch: int = os.cpu_count()) -> List[float]:
    """
    Compute rouge score for generation.

    :param generation_gt: A list of ground truth.
            Must be 2-d list of string.
            Because it can be a multiple ground truth.
    :param generations: A list of generations that LLM generated.
    :param rouge_type: A rouge type to use for evaluation.
        Default is 'RougeL'.
        Choose between rouge1, rouge2, rougeL, and rougeLSum.
        - rouge1: unigram (1-gram) based scoring.
        - rouge2: bigram (2-gram) based scoring.
        - rougeL: Longest Common Subsequence based scoring.
        - rougeLSum: splits text using "\n"
    :param use_stemmer: Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching. This arg is used in the
        DefaultTokenizer, but other tokenizers might or might not choose to
        use this. Default is False.
    :param split_summaries: Whether to add newlines between sentences for rougeLsum.
        Default is False.
    :param batch: The batch size for processing.
        Default is your cpu count.
    :return: A list of computed metric scores.
    """
    rouge_instance = RougeScorer(rouge_types=[rouge_type], use_stemmer=use_stemmer,
                                 split_summaries=split_summaries,
                                 tokenizer=tokenizers.DefaultTokenizer(use_stemmer))

    async def compute(gt: List[str], pred: str) -> float:
        return rouge_instance.score_multi(targets=gt, prediction=pred)[rouge_type].fmeasure

    tasks = [compute(gt, pred) for gt, pred in zip(generation_gt, generations)]
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(process_batch(tasks, batch_size=batch))

    del rouge_instance
    return result


def sem_score(generation_gt: List[List[str]], generations: List[str],
              embedding_model: Optional[BaseEmbedding] = None,
              batch: int = 128) -> List[float]:
    """
    Compute sem score between generation gt and pred with cosine similarity.

    :param generation_gt: A list of ground truth.
            Must be 2-d list of string.
            Because it can be a multiple ground truth.
            It will get the max of cosine similarity between generation_gt and pred.
    :param generations: A list of generations that LLM generated.
    :param embedding_model: Embedding model to use for compute cosine similarity.
        Default is all-mpnet-base-v2 embedding model.
        The paper used this embedding model.
    :param batch: The batch size for processing.
        Default is 128
    :return: A list of computed metric scores.
    """
    if embedding_model is None:
        embedding_model = embedding_models['huggingface_all_mpnet_base_v2']

    embedding_model.embed_batch_size = batch

    openai_embedding_max_length = 8191
    if isinstance(embedding_model, OpenAIEmbedding):
        generations = openai_truncate_by_token(generations, openai_embedding_max_length,
                                               embedding_model.model_name)

    embedded_pred: List[List[float]] = embedding_model.get_text_embedding_batch(generations,
                                                                                show_progress=True)
    gt_lengths = list(map(len, generation_gt))
    flatten_gt = list(itertools.chain.from_iterable(generation_gt))
    if isinstance(embedding_model, OpenAIEmbedding):
        flatten_gt = openai_truncate_by_token(flatten_gt, openai_embedding_max_length,
                                              embedding_model.model_name)
    embedded_gt_flatten = embedding_model.get_text_embedding_batch(flatten_gt, show_progress=True)
    # re-group embedded_gt_flatten with gt_lengths
    iterator = iter(embedded_gt_flatten)
    embedded_gt: List[List[List[float]]] = [list(itertools.islice(iterator, length)) for length in gt_lengths]

    result = []
    for gt, pred in zip(embedded_gt, embedded_pred):
        similarity_scores: List[float] = list(
            map(lambda x: calculate_cosine_similarity(x, pred), gt))
        result.append(max(similarity_scores))

    del embedding_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def g_eval(generation_gt: List[List[str]], generations: List[str],
           metrics: Optional[List[str]] = None,
           model: str = 'gpt-4-0125-preview',
           batch_size: int = 8) -> List[float]:
    """
        Calculate G-Eval score.
        G-eval is a metric that uses high-performance LLM model to evaluate generation performance.
        It evaluates the generation result by coherence, consistency, fluency, and relevance.
        It uses only 'openai' model, and we recommend to use gpt-4 for evaluation accuracy.

        :param generation_gt: A list of ground truth.
            Must be 2-d list of string.
            Because it can be a multiple ground truth.
            It will get the max of g_eval score.
        :param generations: A list of generations that LLM generated.
        :param metrics: A list of metrics to use for evaluation.
            Default is all metrics, which is ['coherence', 'consistency', 'fluency', 'relevance'].
        :param model: OpenAI model name.
            Default is 'gpt-4-0125-preview'.
        :param batch_size: The batch size for processing.
            Default is 8.
        :return: G-Eval score.
    """
    loop = asyncio.get_event_loop()
    tasks = [async_g_eval(gt, pred, metrics, model) for gt, pred in zip(generation_gt, generations)]
    result = loop.run_until_complete(process_batch(tasks, batch_size=batch_size))
    return result


async def async_g_eval(generation_gt: List[str], pred: str,
                       metrics: Optional[List[str]] = None,
                       model: str = 'gpt-4-0125-preview',
                       ) -> float:
    available_metrics = ['coherence', 'consistency', 'fluency', 'relevance']
    if metrics is None:
        metrics = available_metrics
    else:
        assert len(metrics) > 0, "metrics must be a list of string"
        metrics = [metric for metric in metrics if metric in available_metrics]

    current_path = os.path.dirname(os.path.realpath(__file__))
    prompt_path = os.path.join(current_path, 'g_eval_prompts')
    g_eval_prompts = {
        "coherence": open(os.path.join(prompt_path, "coh_detailed.txt")).read(),
        "consistency": open(os.path.join(prompt_path, "con_detailed.txt")).read(),
        "fluency": open(os.path.join(prompt_path, "flu_detailed.txt")).read(),
        "relevance": open(os.path.join(prompt_path, "rel_detailed.txt")).read(),
    }

    client = AsyncOpenAI()

    async def g_eval_score(prompt: str, gen_gt: List[str], pred: str):
        scores = []
        for gt in gen_gt:
            input_prompt = prompt.replace('{{Document}}', gt).replace('{{Summary}}', pred)
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": input_prompt}
                ],
                logprobs=True,
                top_logprobs=5,
                temperature=0,
                max_tokens=2,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                n=20,
            )
            if '(1-3):' in prompt:
                scores.append(get_g_eval_score(response, max_score=3))
            else:
                scores.append(get_g_eval_score(response))
        return max(scores)

    def get_g_eval_score(responses, max_score: int = 5) -> int:
        target_tokens = {str(i): 0 for i in range(1, max_score + 1)}
        for choice in responses.choices:
            first_top_log_probs = choice.logprobs.content[0].top_logprobs
            for i, top_log_prob in enumerate(list(map(lambda x: x.token, first_top_log_probs))):
                if top_log_prob in target_tokens:
                    target_tokens[top_log_prob] += (5 - i)

        return int(max(target_tokens, key=target_tokens.get))

    g_eval_scores = await asyncio.gather(*(g_eval_score(g_eval_prompts[x], generation_gt, pred) for x in metrics))
    return sum(g_eval_scores) / len(g_eval_scores)


def bert_score(generation_gt: List[List[str]], generations: List[str],
               lang: str = 'en',
               batch: int = 128,
               n_threads: int = os.cpu_count()) -> List[float]:
    evaluator = evaluate.load("bertscore")

    df = pd.DataFrame({
        'reference': generation_gt,
        'prediction': generations,
        'lang': lang,
    })

    df = df.explode('reference', ignore_index=False)
    df['bert_score'] = evaluator.compute(predictions=df['prediction'].tolist(),
                                         references=df['reference'].tolist(),
                                         lang=lang, nthreads=n_threads, batch_size=batch)['f1']
    return df.groupby(level=0)['bert_score'].max().tolist()
