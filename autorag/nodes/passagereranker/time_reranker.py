from datetime import datetime
from typing import List, Tuple

from autorag.nodes.passagereranker.base import passage_reranker_node


@passage_reranker_node
def time_reranker(contents_list: List[List[str]],
                  scores_list: List[List[float]], ids_list: List[List[str]],
                  top_k: int, time_list: List[List[datetime]]
                  ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Rerank the passages based on merely the datetime of the passage.
    It uses 'last_modified_datetime' key in the corpus metadata,
    so the metadata should be in the format of {'last_modified_datetime': datetime.datetime} at the corpus data file.

    :param contents_list: The list of lists of contents
    :param scores_list: The list of lists of scores from the initial ranking
    :param ids_list: The list of lists of ids
    :param top_k: The number of passages to be retrieved after reranking
    :param time_list: The metadata list of lists of datetime.datetime
        It automatically extracts the 'last_modified_datetime' key from the metadata in the corpus data.
    :return: The reranked contents, ids, and scores
    """
    def sort_row(contents, scores, ids, time, top_k):
        combined = list(zip(contents, scores, ids, time))
        combined.sort(key=lambda x: x[3], reverse=True)
        sorted_contents, sorted_scores, sorted_ids, _ = zip(*combined)
        return list(sorted_contents)[:top_k], list(sorted_scores)[:top_k], list(sorted_ids)[:top_k]

    reranked_contents, reranked_scores, reranked_ids = zip(
        *map(sort_row, contents_list, scores_list, ids_list, time_list, [top_k] * len(contents_list)))

    return list(reranked_contents), list(reranked_ids), list(reranked_scores)
