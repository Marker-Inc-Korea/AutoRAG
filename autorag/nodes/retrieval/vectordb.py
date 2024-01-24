import asyncio
from typing import List, Tuple

import chromadb
from llama_index.embeddings import BaseEmbedding

from autorag.nodes.retrieval.base import retrieval_node, evenly_distribute_passages


@retrieval_node
def vectordb(queries: List[List[str]], top_k: int, collection: chromadb.Collection,
             embedding_model: BaseEmbedding) -> Tuple[List[List[str]], List[List[float]]]:

    # check if bm25_corpus is valid
    assert (collection.count() > 0), \
        "collection must contain at least one document. Please check you ingested collection correctly."
    # run async vector_db_pure function
    tasks = [vectordb_pure(input_queries, top_k, collection, embedding_model) for input_queries in queries]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    id_result = list(map(lambda x: x[0], results))
    score_result = list(map(lambda x: x[1], results))
    return id_result, score_result


async def vectordb_pure(queries: List[str], top_k: int, collection: chromadb.Collection,
                        embedding_model: BaseEmbedding) -> Tuple[List[str], List[float]]:
    """
    Async VectorDB retrieval function.
    Its usage is for async retrieval of vector_db row by row.

    :param queries: A list of query strings.
    :param top_k: The number of passages to be retrieved.
    :param collection: A chroma collection instance that will be used to retrieve passages.
    :param embedding_model: An embedding model instance that will be used to embed queries.
    :return: A tuple of list of passage ids and list of its scores.
    """
    # embed query
    embedded_queries = (embedding_model.get_query_embedding(query) for query in queries)
    id_result, score_result = [], []
    for embedded_query in embedded_queries:
        result = collection.query(query_embeddings=embedded_query, n_results=top_k)
        id_result.extend(result['ids'])
        score_result.extend(result['distances'])

    # Distribute passages evenly
    id_result, score_result = evenly_distribute_passages(id_result, score_result, top_k)
    # sort id_result and score_result by score
    result = [(_id, score) for score, _id in
              sorted(zip(score_result, id_result), key=lambda pair: pair[0], reverse=True)]
    id_result, score_result = zip(*result)
    return list(id_result), list(score_result)
