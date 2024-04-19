from typing import List

from autorag.nodes.passageaugmenter.base import passage_augmenter_node


@passage_augmenter_node
def pass_passage_augmenter(ids_list: List[List[str]], *args, **kwargs):
    """
    Do not perform augmentation.
    Return given ids as is.
    """
    return ids_list
