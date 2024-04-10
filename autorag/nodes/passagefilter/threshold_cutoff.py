import itertools
from typing import List, Tuple

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding

from autorag.evaluate.metric.util import calculate_cosine_similarity
from autorag.nodes.passagefilter.base import passage_filter_node
from autorag.utils.util import reconstruct_list


@passage_filter_node
def similarity_threshold_cutoff(queries: List[str], contents_list: List[List[str]],
                                scores_list: List[List[float]], ids_list: List[List[str]],
                                threshold: float, embedding_model: BaseEmbedding,
                                batch: int = 128,
                                ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    # Embedding using batch
    embedding_model.embed_batch_size = batch
    query_embeddings = embedding_model.get_text_embedding_batch(queries)

    content_lengths = list(map(len, contents_list))
    content_embeddings_flatten = embedding_model.get_text_embedding_batch(itertools.chain.from_iterable(contents_list))
    content_embeddings = reconstruct_list(content_embeddings_flatten, content_lengths)

    remain_indices = list(map(lambda x: similarity_threshold_cutoff_pure(x[0], x[1], threshold),
                              zip(query_embeddings, content_embeddings)))

    remain_content_list = list(map(lambda c, idx: [c[i] for i in idx], contents_list, remain_indices))
    remain_scores_list = list(map(lambda s, idx: [s[i] for i in idx], scores_list, remain_indices))
    remain_ids_list = list(map(lambda _id, idx: [_id[i] for i in idx], ids_list, remain_indices))

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
