import functools
from typing import List

import pandas as pd
import swifter  # do not delete this line


def retrieval_metric(func):
    @functools.wraps(func)
    def wrapper(retrieval_gt: List[List[str]], ids: List[List[str]]) -> List[float]:
        # make retrieval_gt and ids to pd dataframe
        df = pd.DataFrame({'gt': retrieval_gt, 'pred': ids})
        df[func.__name__] = df.swifter.apply(lambda x: func(x['gt'], x['pred']), axis=1)
        return df[func.__name__].tolist()

    return wrapper


@retrieval_metric
def retrieval_f1(gt: List[str], pred: List[str]):
    recall_score = retrieval_recall.__wrapped__(gt, pred)
    precision_score = retrieval_precision.__wrapped__(gt, pred)
    if recall_score + precision_score == 0:
        return 0
    else:
        return 2 * (recall_score * precision_score) / (recall_score + precision_score)


@retrieval_metric
def retrieval_recall(gt: List[str], pred: List[str]):
    return len(set(gt).intersection(set(pred))) / len(gt)


@retrieval_metric
def retrieval_precision(gt: List[str], pred: List[str]):
    return len(set(gt).intersection(set(pred))) / len(pred)
