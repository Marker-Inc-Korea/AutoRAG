from typing import List

from autorag.nodes.passageaugmenter.base import passage_augmenter_node


@passage_augmenter_node
def pass_passage_augmenter(ids_list: List[List[str]], contents_list: List[List[str]], scores_list: List[List[float]]):
    """
    Do not perform augmentation.
    Return given passages, scores, and ids as is.
    """
    return ids_list, contents_list, scores_list
