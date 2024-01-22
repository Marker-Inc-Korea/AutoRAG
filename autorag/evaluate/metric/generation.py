import functools
from typing import List

import pandas as pd
import sacrebleu


def generation_metric(func):
    @functools.wraps(func)
    def wrapper(generation_gt: List[List[str]], generations: List[str]) -> List[float]:
        # make generation_gt and generations to pd dataframe
        df = pd.DataFrame({'gt': generation_gt, 'pred': generations})
        df[func.__name__] = df.swifter.apply(lambda x: func(x['gt'], x['pred']), axis=1)
        return df[func.__name__].tolist()

    return wrapper


@generation_metric
def bleu(gt: List[str], pred: str) -> float:
    """
    Compute bleu score for generation.

    :param gt: Ground truth.
    :param pred: Prediction.
    :return: The bleu score.
    """
    return sacrebleu.sentence_bleu(pred, gt).score
