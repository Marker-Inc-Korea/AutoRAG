import itertools
from typing import List, Tuple, Optional

import numpy as np
import torch.cuda

from autorag import embedding_models
from autorag.evaluate.metric.util import calculate_cosine_similarity
from autorag.nodes.passagefilter.base import passage_filter_node
from autorag.utils.util import reconstruct_list


@passage_filter_node
def similarity_threshold_cutoff(queries: List[str], contents_list: List[List[str]],
                                scores_list: List[List[float]], ids_list: List[List[str]],
                                threshold: float, embedding_model: Optional[str] = None,
                                batch: int = 128,
                                ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Re-calculate each content's similarity with the query and filter out the contents that are below the threshold.
    If all contents are filtered, keep the only one highest similarity content.
    This is a filter and does not override scores.
    The output of scores is not coming from query-content similarity.

    :param queries: The list of queries to use for filtering
    :param contents_list: The list of lists of contents to filter
    :param scores_list: The list of lists of scores retrieved
    :param ids_list: The list of lists of ids retrieved
    :param threshold: The threshold to cut off
    :param embedding_model: The embedding model to use for calculating similarity
        Default is OpenAIEmbedding.
    :param batch: The number of queries to be processed in a batch
        Default is 128.
    :return: Tuple of lists containing the filtered contents, ids, and scores
    """
    query_embeddings, content_embeddings = embedding_query_content(queries, contents_list, embedding_model, batch)

    remain_indices = list(map(lambda x: similarity_threshold_cutoff_pure(x[0], x[1], threshold),
                              zip(query_embeddings, content_embeddings)))

    remain_content_list = list(map(lambda c, idx: [c[i] for i in idx], contents_list, remain_indices))
    remain_scores_list = list(map(lambda s, idx: [s[i] for i in idx], scores_list, remain_indices))
    remain_ids_list = list(map(lambda _id, idx: [_id[i] for i in idx], ids_list, remain_indices))

    del embedding_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return remain_content_list, remain_ids_list, remain_scores_list


def similarity_threshold_cutoff_pure(query_embedding: str,
                                     content_embeddings: List[List[float]],
                                     threshold: float) -> List[int]:
    """
    Return indices that have to remain.
    Return at least one index if there is nothing to remain.

    :param query_embedding: Query embedding
    :param content_embeddings: Each content embedding
    :param threshold: The threshold to cut off
    :return: Indices to remain at the contents
    """

    similarities = np.array(list(map(lambda x: calculate_cosine_similarity(query_embedding, x),
                                     content_embeddings)))
    result = np.where(similarities >= threshold)[0].tolist()
    if len(result) > 0:
        return result
    return [np.argmax(similarities)]


def embedding_query_content(queries: List[str], contents_list: List[List[str]],
                            embedding_model: Optional[str] = None, batch: int = 128):
    if embedding_model is None:
        embedding_model = embedding_models['openai']
    else:
        embedding_model = embedding_models[embedding_model]

    # Embedding using batch
    embedding_model.embed_batch_size = batch
    query_embeddings = embedding_model.get_text_embedding_batch(queries)

    content_lengths = list(map(len, contents_list))
    content_embeddings_flatten = embedding_model.get_text_embedding_batch(list(
        itertools.chain.from_iterable(contents_list)))
    content_embeddings = reconstruct_list(content_embeddings_flatten, content_lengths)
    return query_embeddings, content_embeddings
