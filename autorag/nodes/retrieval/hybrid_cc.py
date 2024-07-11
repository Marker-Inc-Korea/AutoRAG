from typing import Tuple, List

import numpy as np
import pandas as pd

from autorag.nodes.retrieval import retrieval_node


def normalize_mm(scores: List[str],
                 fixed_min_value: float = 0):
    arr = np.array(scores)
    max_value = np.max(arr)
    min_value = np.min(arr)
    norm_score = (arr - min_value) / (max_value - min_value)
    return norm_score


# def normalize_tmm(semantic_scores: List[float], lexical_scores: List[float],
#                   semantic_theoretical_min_value: float = -1.0,
#                   lexical_theoretical_min_value: float = 0.0):
#     concat_arr = np.array(semantic_scores + lexical_scores)
#     max_score = np.max(concat_arr)


normalize_method_dict = {
    'mm': normalize_mm,
    'tmm': normalize_tmm,
    # 'z': normalize_z,
    # 'dbsf': normalize_dbsf,
}


@retrieval_node
def hybrid_cc(
        ids: Tuple,
        scores: Tuple,
        top_k: int,
        # TODO: add metrics and strategy to evaluate retrieval.
        # TODO: add fixed weight for Runner.run
        # TODO: how to save selected weight value?
        normalize_method: str,
        weight_range: tuple = (0.0, 1.0),
        test_weight_size: int = 100,
        semantic_theoretical_min_value: float = -1.0,
        lexical_theoretical_min_value: float = 0.0,
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Hybrid CC function.
    CC (convex combination) is a method to fuse lexical and semantic retrieval results.
    It is a method that first normalizes the scores of each retrieval result,
    and then combines them with the given weights.
    It is uniquer than other retrieval modules, because it does not really execute retrieval,
    but just fuse the results of other retrieval functions.
    So you have to run more than two retrieval modules before running this function.
    And collect ids and scores result from each retrieval module.
    Make it as tuple and input it to this function.

    :param ids: The tuple of ids that you want to fuse.
        The length of this must be the same as the length of scores.
        The semantic retrieval ids must be the first index.
    :param scores: The retrieve scores that you want to fuse.
        The length of this must be the same as the length of ids.
        The semantic retrieval scores must be the first index.
    :param top_k: The number of passages to be retrieved.
    :param normalize_method: The normalization method to use.
      There are some normalization method that you can use at the hybrid cc method.
      AutoRAG support following.
        - `mm`: Min-max scaling
        - `tmm`: Theoretical min-max scaling
        - `z`: z-score normalization
        - `dbsf`: 3-sigma normalization
    :param weight_range: The range of the weight that you want to explore. If the weight is 1.0, it means the
      weight to the semantic module will be 1.0 and weight to the lexical module will be 0.0.
      You have to input this value as tuple. It looks like this. `(0.2, 0.8)`. Default is `(0.0, 1.0)`.
    :param test_weight_size: The size of the weight that tested for optimization. If the weight range
      is `(0.2, 0.8)` and the size is 6, it will evaluate the following weights.
      `0.2, 0.3, 0.4, 0.5, 0.6, 0.7`. Default is 100.
  :param semantic_theoretical_min_value: This value used by `tmm` normalization method. You can set the
    theoretical minimum value by yourself. Default is -1.
    :param lexical_theoretical_min_value: This value used by `tmm` normalization method. You can set the
        theoretical minimum value by yourself. Default is 0.
    :return: The tuple of ids and fused scores that fused by CC.
    """
    assert len(ids) == len(scores), "The length of ids and scores must be the same."
    assert len(ids) > 1, "You must input more than one retrieval results."
    assert top_k > 0, "top_k must be greater than 0."
    assert weight_range[0] < weight_range[1], "The smaller range must be at the first of the tuple"
    assert weight_range[0] >= 0, "The range must be greater than 0."
    assert weight_range[1] <= 1, "The range must be less than 1."

    id_df = pd.DataFrame({f'id_{i}': id_list for i, id_list in enumerate(ids)})
    score_df = pd.DataFrame({f'score_{i}': score_list for i, score_list in enumerate(scores)})
    df = pd.concat([id_df, score_df], axis=1)

    def cc_pure_apply(row):
        ids_tuple = tuple(row[[f'id_{i}' for i in range(len(ids))]].values)
        scores_tuple = tuple(row[[f'score_{i}' for i in range(len(scores))]].values)
        return pd.Series(cc_pure(ids_tuple, scores_tuple, (0.2, 0.8), top_k))

    df[['cc_id', 'cc_score']] = df.apply(cc_pure_apply, axis=1)
    return df['cc_id'].tolist(), df['cc_score'].tolist()


def fuse_per_query(semantic_ids: List[str], lexical_ids: List[str],
                   semantic_scores: List[float], lexical_scores: List[float],
                   normalize_method: str,
                   weight: float,
                   top_k: int,
                   semantic_theoretical_min_value: float,
                   lexical_theoretical_min_value: float):
    normalize_func = normalize_method_dict[normalize_method]
    norm_semantic_scores = normalize_func(semantic_scores, semantic_theoretical_min_value)
    norm_lexical_scores = normalize_func(lexical_scores, lexical_theoretical_min_value)
    ids = [semantic_ids, lexical_ids]
    scores = [norm_semantic_scores, norm_lexical_scores]
    df = pd.concat([pd.Series(dict(zip(_id, score))) for _id, score in zip(ids, scores)], axis=1)
    df.columns = ['semantic', 'lexical']
    df = df.fillna(0)
    df['weighted_sum'] = df.mul((weight, 1.0 - weight)).sum(axis=1)
    df = df.sort_values(by='weighted_sum', ascending=False)
    return df.index.tolist()[:top_k], df['weighted_sum'][:top_k].tolist()
