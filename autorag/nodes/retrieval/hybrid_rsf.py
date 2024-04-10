from collections import defaultdict
from typing import List, Tuple, Optional

import pandas as pd

from autorag.nodes.retrieval.base import retrieval_node


@retrieval_node
def hybrid_rsf(ids: Tuple, scores: Tuple, top_k: int,
               dist_based: Optional[bool] = False,
               weights: Tuple = (0.5, 0.5)) -> Tuple[List[str], List[float]]:
    assert len(ids) == len(scores), "The length of ids and scores must be the same."
    assert len(ids) > 1, "You must input more than one retrieval results."
    assert top_k > 0, "top_k must be greater than 0."
    assert sum(weights) == 1, "The sum of weights must be 1."

    # Initialize DataFrame for ids and scores
    id_df = pd.DataFrame({f'id_{i}': id_list for i, id_list in enumerate(ids)})
    score_df = pd.DataFrame({f'score_{i}': score_list for i, score_list in enumerate(scores)})
    df = pd.concat([id_df, score_df], axis=1)

    # Apply relative score fusion
    def rsf_pure_apply(row):
        ids_tuple = tuple(row[[f'id_{i}' for i in range(len(ids))]].values)
        scores_tuple = tuple(row[[f'score_{i}' for i in range(len(scores))]].values)
        return pd.Series(rsf_pure(ids_tuple, scores_tuple, top_k, dist_based, weights))

    df[['rsf_id', 'rsf_score']] = df.apply(rsf_pure_apply, axis=1)
    return df['rsf_id'].tolist(), df['rsf_score'].tolist()


def rsf_pure(ids: Tuple, scores: Tuple, top_k: int,
             dist_based: Optional[bool] = False, weights: Tuple = (0.5, 0.5)) -> Tuple[List[str], List[float]]:
    # Initialize min and max scores for scaling
    min_max_scores = {}
    for i, score_list in enumerate(scores):
        if not score_list:
            min_max_scores[i] = (0.0, 0.0)
            continue
        if dist_based:
            mean_score = sum(score_list) / len(score_list)
            std_dev = (sum((x - mean_score) ** 2 for x in score_list) / len(score_list)) ** 0.5
            min_score = mean_score - 3 * std_dev
            max_score = mean_score + 3 * std_dev
        else:
            min_score = min(score_list)
            max_score = max(score_list)
        min_max_scores[i] = (min_score, max_score)

    # Scale scores and apply weights
    scaled_scores = []
    for i, (score_list, weight) in enumerate(zip(scores, weights)):
        scaled_score_list = []
        min_score, max_score = min_max_scores[i]
        for score in score_list:
            if max_score == min_score:
                scaled_score = 1.0 if max_score > 0 else 0.0
            else:
                scaled_score = (score - min_score) / (max_score - min_score)
            scaled_score *= weight
            scaled_score_list.append(scaled_score)
        scaled_scores.append(scaled_score_list)

    # Fuse scores
    fused_scores_dict = defaultdict(float)
    for score_list, id_list in zip(scaled_scores, ids):
        for score, doc_id in zip(score_list, id_list):
            fused_scores_dict[doc_id] += score

    # Sort by fused scores and select top_k
    sorted_items = sorted(fused_scores_dict.items(), key=lambda item: item[1], reverse=True)[:top_k]
    top_ids, top_scores = zip(*sorted_items) if sorted_items else ([], [])

    return list(top_ids), list(top_scores)
