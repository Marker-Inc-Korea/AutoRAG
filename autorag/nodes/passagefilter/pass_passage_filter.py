from typing import List

from autorag.nodes.passagefilter.base import passage_filter_node


@passage_filter_node
def pass_passage_filter(queries: List[str], contents_list: List[List[str]],
                        scores_list: List[List[float]], ids_list: List[List[str]]):
    """
    Do not perform filtering.
    Return given passages, scores, and ids as is.
    """
    return contents_list, ids_list, scores_list
