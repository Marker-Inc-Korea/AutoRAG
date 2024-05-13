from typing import List, Tuple, Optional

import numpy as np
import torch.cuda

from autorag.evaluate.metric.util import calculate_cosine_similarity
from autorag.nodes.passagefilter.base import passage_filter_node
from autorag.nodes.passagefilter.similarity_threshold_cutoff import embedding_query_content


@passage_filter_node
def similarity_percentile_cutoff(queries: List[str], contents_list: List[List[str]],
                                 scores_list: List[List[float]], ids_list: List[List[str]],
                                 percentile: float, embedding_model: Optional[str] = None,
                                 batch: int = 128,
                                 ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Re-calculate each content's similarity with the query and filter out the contents that are below the content's
    length times percentile. If This is a filter and does not override scores. The output of scores is not coming from
    query-content similarity.
    If the value of content's length times percentile is less than 1, keep the only one highest similarity content.

    :param queries: The list of queries to use for filtering
    :param contents_list: The list of lists of contents to filter
    :param scores_list: The list of lists of scores retrieved
    :param ids_list: The list of lists of ids retrieved
    :param percentile: The percentile to cut off
    :param embedding_model: The embedding model to use for calculating similarity
        Default is OpenAIEmbedding.
    :param batch: The number of queries to be processed in a batch
        Default is 128.
    :return: Tuple of lists containing the filtered contents, ids, and scores
    """
    query_embeddings, content_embeddings = embedding_query_content(queries, contents_list, embedding_model, batch)

    results = list(map(lambda x: similarity_percentile_cutoff_pure(x[0], x[1], x[2], x[3], x[4], percentile),
                       zip(query_embeddings, content_embeddings, contents_list, ids_list, scores_list)))

    remain_content_list = list(map(lambda x: x[0], results))
    remain_ids_list = list(map(lambda x: x[1], results))
    remain_scores_list = list(map(lambda x: x[2], results))

    del embedding_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return remain_content_list, remain_ids_list, remain_scores_list


def similarity_percentile_cutoff_pure(query_embedding: str,
                                      content_embeddings: List[List[float]],
                                      content_list: List[str],
                                      ids_list: List[str],
                                      scores_list: List[float],
                                      percentile: float) -> Tuple[List[str], List[str], List[float]]:
    """
    Return tuple of lists containing the filtered contents, ids, and scores

    :param query_embedding: Query embedding
    :param content_embeddings: Each content embedding
    :param content_list: Each content
    :param ids_list: Each id
    :param scores_list: Each score
    :param percentile: The percentile to cut off
    :return: Tuple of lists containing the filtered contents, ids, and scores
    """
    num_top_k = int(len(content_embeddings) * percentile)

    if num_top_k == 0:
        num_top_k = 1

    similarities = np.array(list(map(lambda x: calculate_cosine_similarity(query_embedding, x),
                                     content_embeddings))).tolist()

    content_id_score_similarity = list(zip(ids_list, content_list, scores_list, similarities))

    sorted_content_id_score_similarity = sorted(content_id_score_similarity, key=lambda x: x[3], reverse=True)[
                                         :num_top_k]

    content_result, id_result, score_result, _ = zip(*sorted_content_id_score_similarity)
    return list(content_result), list(id_result), list(score_result)
