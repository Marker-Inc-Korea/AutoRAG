import functools
from typing import List, Optional

import evaluate
import pandas as pd
import sacrebleu
from llama_index.core.embeddings import BaseEmbedding

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
        df = pd.DataFrame({'gt': generation_gt, 'pred': generations})
        df[func.__name__] = df.swifter.apply(lambda x: func(x['gt'], x['pred'], **kwargs), axis=1)
        return df[func.__name__].tolist()

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

    df = pd.DataFrame({'gt': generation_gt, 'pred': generations})
    df[key] = df.swifter.apply(lambda x: compute_score(x['gt'], x['pred']), axis=1)
    return df[key].tolist()


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
    return huggingface_evaluate(meteor_instance, 'meteor', generation_gt, generations)


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
    return huggingface_evaluate(rouge_instance, 'rougeL', generation_gt, generations)


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
