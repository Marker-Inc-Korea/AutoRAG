from datetime import datetime
from typing import List, Tuple

from autorag.nodes.passagereranker.base import passage_reranker_node


@passage_reranker_node
def time_reranker(contents_list: List[List[str]],
                  scores_list: List[List[float]], ids_list: List[List[str]],
                  top_k: int, time_list: List[List[datetime]]
                  ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    def sort_row(contents, scores, ids, time, top_k):
        combined = list(zip(contents, scores, ids, time))
        combined.sort(key=lambda x: x[3], reverse=True)
        sorted_contents, sorted_scores, sorted_ids, _ = zip(*combined)
        return list(sorted_contents)[:top_k], list(sorted_scores)[:top_k], list(sorted_ids)[:top_k]

    reranked_contents, reranked_scores, reranked_ids = zip(
        *map(sort_row, contents_list, scores_list, ids_list, time_list, [top_k] * len(contents_list)))

    return list(reranked_contents), list(reranked_ids), list(reranked_scores)
