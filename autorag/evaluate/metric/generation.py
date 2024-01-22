import functools
from typing import List

import evaluate
import pandas as pd
import sacrebleu


def generation_metric(func):
    @functools.wraps(func)
    def wrapper(generation_gt: List[List[str]], generations: List[str]) -> List[float]:
        """
        Compute generation metric.

        :param generation_gt: A list of ground truth.
            Must be 2-d list of string.
            Because it can be a multiple ground truth.
        :param generations: A list of generations that LLM generated.
        :return: A list of computed metric scores.
        """
        # make generation_gt and generations to pd dataframe
        df = pd.DataFrame({'gt': generation_gt, 'pred': generations})
        df[func.__name__] = df.swifter.apply(lambda x: func(x['gt'], x['pred']), axis=1)
        return df[func.__name__].tolist()

    return wrapper


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

    def compute_meteor(gt: List[str], pred: str) -> float:
        return max(list(map(
            lambda x: meteor_instance.compute(predictions=[pred], references=[x])['meteor'], gt)))

    df = pd.DataFrame({'gt': generation_gt, 'pred': generations})
    df['meteor'] = df.swifter.apply(lambda x: compute_meteor(x['gt'], x['pred']), axis=1)
    return df['meteor'].tolist()
