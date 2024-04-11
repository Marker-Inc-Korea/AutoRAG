from datetime import datetime
from typing import List, Tuple

from autorag.nodes.passagefilter.base import passage_filter_node


@passage_filter_node
def time_filter(contents_list: List[List[str]],
                scores_list: List[List[float]], ids_list: List[List[str]],
                time_list: List[List[datetime]],
                threshold: str,
                ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    def sort_row(contents, scores, ids, time, datetime_threshold):
        combined_filtered = [(content, score, id, t) for content, score, id, t in zip(contents, scores, ids, time) if
                             t >= datetime_threshold]
        if combined_filtered:
            remain_contents, remain_scores, remain_ids, _ = zip(*combined_filtered)
            return list(remain_contents), list(remain_ids), list(remain_scores)
        else:
            return [], [], []

    # threshold 유효성 검사 추가하기
    datetime_threshold = datetime.strptime(threshold, "%Y-%m-%d")
    remain_contents_list, remain_ids_list, remain_scores_list = zip(
        *map(sort_row, contents_list, scores_list, ids_list, time_list, [datetime_threshold] * len(contents_list)))

    return remain_contents_list, remain_ids_list, remain_scores_list
