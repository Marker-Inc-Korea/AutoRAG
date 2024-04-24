import logging
from datetime import datetime, date
from typing import List, Tuple, Union

from autorag.nodes.passagefilter.base import passage_filter_node

logger = logging.getLogger("AutoRAG")


@passage_filter_node
def recency_filter(contents_list: List[List[str]],
                   scores_list: List[List[float]], ids_list: List[List[str]],
                   time_list: List[List[datetime]],
                   threshold: Union[datetime, date],
                   ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Filter out the contents that are below the threshold datetime.
    If all contents are filtered, keep the only one recency content.
    If the threshold date format is incorrect, return the original contents.

    :param contents_list: The list of lists of contents to filter
    :param scores_list: The list of lists of scores retrieved
    :param ids_list: The list of lists of ids retrieved
    :param time_list: The list of lists of datetime retrieved
    :param threshold: The threshold to cut off
    :return: Tuple of lists containing the filtered contents, ids, and scores
    """
    if not (isinstance(threshold, datetime) or isinstance(threshold, date)):
        raise ValueError(f"Threshold should be a datetime object, but got {type(threshold)}")

    if not isinstance(threshold, datetime):
        threshold = datetime.combine(threshold, datetime.min.time())

    time_list = [
        list(map(lambda t: datetime.combine(t, datetime.min.time()) if not isinstance(t, datetime) else t, time))
        for time in time_list
    ]

    def sort_row(contents, scores, ids, time, _datetime_threshold):
        combined = list(zip(contents, scores, ids, time))
        combined_filtered = [item for item in combined if item[3] >= _datetime_threshold]

        if combined_filtered:
            remain_contents, remain_scores, remain_ids, _ = zip(*combined_filtered)
        else:
            combined.sort(key=lambda x: x[3], reverse=True)
            remain_contents, remain_scores, remain_ids, _ = zip(*combined[:1])

        return list(remain_contents), list(remain_ids), list(remain_scores)

    remain_contents_list, remain_ids_list, remain_scores_list = zip(
        *map(sort_row, contents_list, scores_list, ids_list, time_list, [threshold] * len(contents_list)))

    return remain_contents_list, remain_ids_list, remain_scores_list
