from typing import List

from autorag.nodes.passagereranker.base import passage_reranker_node


@passage_reranker_node
def pass_reranker(queries: List[str], contents_list: List[List[str]],
                  scores_list: List[List[float]], ids_list: List[List[str]],
                  top_k: int):
    """
    Do not perform reranking.
    Return the given top-k passages as is.
    """
    contents_list = list(map(lambda x: x[:top_k], contents_list))
    scores_list = list(map(lambda x: x[:top_k], scores_list))
    ids_list = list(map(lambda x: x[:top_k], ids_list))
    return contents_list, ids_list, scores_list
