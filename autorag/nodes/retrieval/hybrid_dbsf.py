from typing import List, Tuple

import pandas as pd

from autorag.nodes.retrieval.base import retrieval_node
from autorag.nodes.retrieval.hybrid_rsf import score_fusion


@retrieval_node
def hybrid_dbsf(ids: Tuple, scores: Tuple, top_k: int,
                weights: Tuple = (0.5, 0.5)) -> Tuple[List[str], List[float]]:
    """
    Hybrid DBSF function.
    DBSF (Distribution Based Score Fusion) is a method to combine multiple retrieval results based on their distribution.
    It is uniquer than other retrieval modules, because it does not really execute retrieval,
    but just fuse the results of other retrieval functions.
    So you have to run more than two retrieval modules before running this function.
    And collect ids and scores result from each retrieval module.
    Make it as tuple and input it to this function.

    :param ids: The tuple of ids that you want to fuse.
        The length of this must be the same as the length of scores.
    :param scores: The retrieve scores that you want to fuse.
        The length of this must be the same as the length of ids.
    :param top_k: The number of passages to be retrieved.
    :param weights: Weight for each retrieval result.
        Default is (0.5, 0.5).
        You must set its length as the same as the length of ids and scores.
        Plus, the sum of the weights must be 1.
    :return: The tuple of ids and fused scores that fused by DBSF.
    """
    assert len(ids) == len(scores), "The length of ids and scores must be the same."
    assert len(ids) > 1, "You must input more than one retrieval results."
    assert top_k > 0, "top_k must be greater than 0."
    assert sum(weights) == 1, "The sum of weights must be 1."

    # Initialize DataFrame for ids and scores
    id_df = pd.DataFrame({f'id_{i}': id_list for i, id_list in enumerate(ids)})
    score_df = pd.DataFrame({f'score_{i}': score_list for i, score_list in enumerate(scores)})
    df = pd.concat([id_df, score_df], axis=1)

    # Apply relative score fusion
    def dbsf_pure_apply(row):
        ids_tuple = tuple(row[[f'id_{i}' for i in range(len(ids))]].values)
        scores_tuple = tuple(row[[f'score_{i}' for i in range(len(scores))]].values)
        return pd.Series(dbsf_pure(ids_tuple, scores_tuple, top_k, weights))

    df[['dbsf_id', 'dbsf_score']] = df.apply(dbsf_pure_apply, axis=1)
    return df['dbsf_id'].tolist(), df['dbsf_score'].tolist()


def dbsf_pure(ids: Tuple, scores: Tuple, top_k: int,
              weights: Tuple = (0.5, 0.5)) -> Tuple[List[str], List[float]]:
    min_max_scores = dbsf_min_max_scores(scores)
    top_ids, top_scores = score_fusion(ids, scores, top_k, min_max_scores, weights)
    return top_ids, top_scores


def dbsf_min_max_scores(scores: Tuple):
    min_max_scores = {}
    for i, score_list in enumerate(scores):
        if not score_list:
            min_max_scores[i] = (0.0, 0.0)
            continue
        mean_score = sum(score_list) / len(score_list)
        std_dev = (sum((x - mean_score) ** 2 for x in score_list) / len(score_list)) ** 0.5
        min_score = mean_score - 3 * std_dev
        max_score = mean_score + 3 * std_dev
        min_max_scores[i] = (min_score, max_score)
    return min_max_scores
