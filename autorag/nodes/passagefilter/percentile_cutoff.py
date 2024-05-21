from typing import List, Tuple

import pandas as pd

from autorag.nodes.passagefilter.base import passage_filter_node
from autorag.utils.util import sort_by_scores, select_top_k


@passage_filter_node
def percentile_cutoff(queries: List[str], contents_list: List[List[str]],
                      scores_list: List[List[float]], ids_list: List[List[str]],
                      percentile: float, reverse: bool = False,
                      ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Filter out the contents that are below the content's length times percentile.
    If This is a filter and does not override scores.
    If the value of content's length times percentile is less than 1, keep the only one highest similarity content.

    :param queries: The list of queries to use for filtering
    :param contents_list: The list of lists of contents to filter
    :param scores_list: The list of lists of scores retrieved
    :param ids_list: The list of lists of ids retrieved
    :param percentile: The percentile to cut off
    :param reverse: If True, the lower the score, the better
        Default is False.
    :return: Tuple of lists containing the filtered contents, ids, and scores
    """
    num_top_k = max(1, int(len(scores_list[0]) * percentile))

    df = pd.DataFrame({
        'contents': contents_list,
        'ids': ids_list,
        'scores': scores_list,
    })

    reverse = not reverse
    df[['contents', 'ids', 'scores']] = df.apply(sort_by_scores, axis=1, result_type='expand', reverse=reverse)
    results = select_top_k(df, ['contents', 'ids', 'scores'], num_top_k)

    return results['contents'].tolist(), results['ids'].tolist(), results['scores'].tolist()
