from typing import Tuple, List

import pandas as pd

from autorag.nodes.retrieval import retrieval_node


@retrieval_node
def hybrid_cc(
        ids: Tuple,
        scores: Tuple,
        top_k: int,
        weights: Tuple = (0.5, 0.5)) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Hybrid CC function.
    CC (convex combination) is a method to fuse multiple retrieval results.
    It is a method that first normalizes the scores of each retrieval result,
    and then combines them with the given weights.
    To use this function, you must input ids and scores as tuple.
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
    :return: The tuple of ids and fused scores that fused by CC.
    """
    assert len(ids) == len(scores), "The length of ids and scores must be the same."
    assert len(ids) == len(weights), "The length of weights must be the same as the length of ids."
    assert len(ids) > 1, "You must input more than one retrieval results."
    assert top_k > 0, "top_k must be greater than 0."
    assert sum(weights) == 1, "The sum of weights must be 1."

    id_df = pd.DataFrame({f'id_{i}': id_list for i, id_list in enumerate(ids)})
    score_df = pd.DataFrame({f'score_{i}': score_list for i, score_list in enumerate(scores)})
    df = pd.concat([id_df, score_df], axis=1)

    def cc_pure_apply(row):
        ids_tuple = tuple(row[[f'id_{i}' for i in range(len(ids))]].values)
        scores_tuple = tuple(row[[f'score_{i}' for i in range(len(scores))]].values)
        return pd.Series(cc_pure(ids_tuple, scores_tuple, weights, top_k))

    df[['cc_id', 'cc_score']] = df.apply(cc_pure_apply, axis=1)
    return df['cc_id'].tolist(), df['cc_score'].tolist()


def cc_pure(ids: Tuple, scores: Tuple, weights: Tuple, top_k: int) -> Tuple[
    List[str], List[float]]:
    df = pd.concat([pd.Series(dict(zip(_id, score))) for _id, score in zip(ids, scores)], axis=1)
    normalized_scores = (df - df.min()) / (df.max() - df.min())
    normalized_scores = normalized_scores.fillna(0)
    normalized_scores['weighted_sum'] = normalized_scores.mul(weights).sum(axis=1)
    normalized_scores = normalized_scores.sort_values(by='weighted_sum', ascending=False)
    return normalized_scores.index.tolist()[:top_k], normalized_scores['weighted_sum'][:top_k].tolist()
