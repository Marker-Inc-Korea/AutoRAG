from typing import List, Tuple

import numpy as np

from autorag.nodes.passagefilter.base import passage_filter_node


@passage_filter_node
def threshold_cutoff(queries: List[str], contents_list: List[List[str]],
                     scores_list: List[List[float]], ids_list: List[List[str]],
                     threshold: float, reverse: bool = False,
                     ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Filters the contents, scores, and ids based on a previous result's score.
    Keeps at least one item per query if all scores are below the threshold.

    :param queries: List of query strings (not used in the current implementation).
    :param contents_list: List of content strings for each query.
    :param scores_list: List of scores for each content.
    :param ids_list: List of ids for each content.
    :param threshold: The minimum score to keep an item.
    :param reverse: If True, the lower the score, the better.
        Default is False.
    :return: Filtered lists of contents, ids, and scores.
    """
    remain_indices = list(map(lambda x: threshold_cutoff_pure(x, threshold, reverse), scores_list))

    remain_content_list = list(map(lambda c, idx: [c[i] for i in idx], contents_list, remain_indices))
    remain_scores_list = list(map(lambda s, idx: [s[i] for i in idx], scores_list, remain_indices))
    remain_ids_list = list(map(lambda _id, idx: [_id[i] for i in idx], ids_list, remain_indices))

    return remain_content_list, remain_ids_list, remain_scores_list


def threshold_cutoff_pure(scores_list: List[float],
                          threshold: float,
                          reverse: bool = False) -> List[int]:
    """
    Return indices that have to remain.
    Return at least one index if there is nothing to remain.

    :param scores_list: Each score
    :param threshold: The threshold to cut off
    :param reverse: If True, the lower the score, the better
        Default is False.
    :return: Indices to remain at the contents
    """
    if isinstance(scores_list, np.ndarray):
        if reverse:
            remain_indices = np.where(scores_list <= threshold)[0]
            default_index = np.argmin(scores_list)
        else:
            remain_indices = np.where(scores_list >= threshold)[0]
            default_index = np.argmax(scores_list)
    else:
        if reverse:
            remain_indices = [i for i, score in enumerate(scores_list) if score <= threshold]
            default_index = scores_list.index(min(scores_list))
        else:
            remain_indices = [i for i, score in enumerate(scores_list) if score >= threshold]
            default_index = scores_list.index(max(scores_list))

    return remain_indices.tolist() if isinstance(remain_indices, np.ndarray) and remain_indices.size > 0 else \
        remain_indices if remain_indices else [default_index]
