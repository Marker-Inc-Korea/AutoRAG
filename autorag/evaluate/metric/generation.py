import functools
import os
from typing import List, Optional

import evaluate
import sacrebleu
from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI

from autorag import embedding_models
from autorag.evaluate.metric.util import calculate_cosine_similarity


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
                         generation_gt: List[List[str]], generations: List[str]) -> List[float]:
    """
    Compute huggingface evaluate metric.

    :param instance: The instance of huggingface evaluates metric.
    :param key: The key to retrieve result score from huggingface evaluate result.
    :param generation_gt: A list of ground truth.
        Must be 2-d list of string.
    :param generations: A list of generations that LLM generated.
    :return: The list of scores.
    """

    def compute_score(gt: List[str], pred: str) -> float:
        return max(list(map(
            lambda x: instance.compute(predictions=[pred], references=[x])[key], gt)))

    result = list(map(lambda x: compute_score(x[0], x[1]), zip(generation_gt, generations)))
    return result


@generation_metric
def bleu(gt: List[str], pred: str) -> float:
    """
    Compute bleu score for generation.
    """
    return sacrebleu.sentence_bleu(pred, gt).score


def meteor(generation_gt: List[List[str]], generations: List[str]) -> List[float]:
    """
    Compute meteor score for generation.

    :param generation_gt: A list of ground truth.
            Must be 2-d list of string.
            Because it can be a multiple ground truth.
    :param generations: A list of generations that LLM generated.
    :return: A list of computed metric scores.
    """
    meteor_instance = evaluate.load("meteor")
    result = huggingface_evaluate(meteor_instance, 'meteor', generation_gt, generations)
    del meteor_instance
    return result


def rouge(generation_gt: List[List[str]], generations: List[str]) -> List[float]:
    """
    Compute rouge score for generation.

    :param generation_gt: A list of ground truth.
            Must be 2-d list of string.
            Because it can be a multiple ground truth.
    :param generations: A list of generations that LLM generated.
    :return: A list of computed metric scores.
    """
    rouge_instance = evaluate.load("rouge")
    result = huggingface_evaluate(rouge_instance, 'rougeL', generation_gt, generations)
    del rouge_instance
    return result


@generation_metric
def sem_score(generation_gt: List[str], pred: str, embedding_model: Optional[BaseEmbedding] = None) -> float:
    """
    Compute sem score between generation gt and pred with cosine similarity.

    :param generation_gt: A list of ground truth.
        Must be list of string.
        It will get the max of cosine similarity between generation_gt and pred.
    :param pred: Model prediction.
    :param embedding_model: Embedding model to use for compute cosine similarity.
        Default is all-mpnet-base-v2 embedding model.
        The paper used this embedding model.
    :return: Sem score between generation_gt and pred.
    """
    if embedding_model is None:
        embedding_model = embedding_models['huggingface_all_mpnet_base_v2']

    gt_embeddings = embedding_model.get_text_embedding_batch(generation_gt)
    pred_embedding = embedding_model.get_text_embedding(pred)

    # calculate cosine similarity
    similarity_scores: List[float] = list(map(lambda x: calculate_cosine_similarity(x, pred_embedding), gt_embeddings))
    return max(similarity_scores)


@generation_metric
def g_eval(generation_gt: List[str], pred: str,
           metrics: Optional[List[str]] = None,
           model: str = 'gpt-4-0125-preview',
           ) -> float:
    """
    Calculate G-Eval score.
    G-eval is a metric that uses high-performance LLM model to evaluate generation performance.
    It evaluates the generation result by coherence, consistency, fluency, and relevance.
    It uses only 'openai' model, and we recommend to use gpt-4 for evaluation accuracy.

    :param generation_gt: A list of ground truth.
    :param pred: Model generation.
    :param metrics: A list of metrics to use for evaluation.
        Default is all metrics, which is ['coherence', 'consistency', 'fluency', 'relevance'].
    :param model: OpenAI model name.
        Default is 'gpt-4-0125-preview'.
    :return: G-Eval score.
    """
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

    client = OpenAI()

    def g_eval_score(prompt: str, gen_gt: List[str], pred: str):
        scores = []
        for gt in gen_gt:
            input_prompt = prompt.replace('{{Document}}', gt).replace('{{Summary}}', pred)
            response = client.chat.completions.create(
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

    g_eval_scores = list(map(lambda x: g_eval_score(g_eval_prompts[x], generation_gt, pred), metrics))
    return sum(g_eval_scores) / len(g_eval_scores)
