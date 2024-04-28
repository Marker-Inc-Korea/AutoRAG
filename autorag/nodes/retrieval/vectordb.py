import asyncio
from typing import List, Tuple

import chromadb
import pandas as pd
from chromadb.utils.batch_utils import create_batches
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.nodes.retrieval.base import retrieval_node, evenly_distribute_passages
from autorag.utils import validate_corpus_dataset
from autorag.utils.util import process_batch, openai_truncate_by_token


@retrieval_node
def vectordb(queries: List[List[str]], top_k: int, collection: chromadb.Collection,
             embedding_model: BaseEmbedding,
             batch: int = 128) -> Tuple[List[List[str]], List[List[float]]]:
    """
    VectorDB retrieval function.
    You have to get a chroma collection that is already ingested.
    You have to get an embedding model that is already used in ingesting.

    :param queries: 2-d list of query strings.
        Each element of the list is a query strings of each row.
    :param top_k: The number of passages to be retrieved.
    :param collection: A chroma collection instance that will be used to retrieve passages.
    :param embedding_model: An embedding model instance that will be used to embed queries.
    :param batch: The number of queries to be processed in parallel.
        This is used to prevent API error at the query embedding.
        Default is 128.

    :return: The 2-d list contains a list of passage ids that retrieved from vectordb and 2-d list of its scores.
        It will be a length of queries. And each element has a length of top_k.
    """
    # check if bm25_corpus is valid
    assert (collection.count() > 0), \
        "collection must contain at least one document. Please check you ingested collection correctly."
    # run async vector_db_pure function
    tasks = [vectordb_pure(input_queries, top_k, collection, embedding_model) for input_queries in queries]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_batch(tasks, batch_size=batch))
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

    :return: The tuple contains a list of passage ids that retrieved from vectordb and a list of its scores.
    """
    # embed query
    embedded_queries = list(map(embedding_model.get_query_embedding, queries))

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


def vectordb_ingest(collection: chromadb.Collection, corpus_data: pd.DataFrame, embedding_model: BaseEmbedding,
                    batch: int = 128):
    embedding_model.embed_batch_size = batch
    validate_corpus_dataset(corpus_data)
    ids = corpus_data['doc_id'].tolist()

    # Query the collection to check if IDs already exist
    existing_ids = set(collection.get(ids=ids)['ids'])  # Assuming 'ids' is the key in the response
    new_passage = corpus_data[~corpus_data['doc_id'].isin(existing_ids)]

    if not new_passage.empty:
        new_contents = new_passage['contents'].tolist()

        # truncate by token if embedding_model is OpenAIEmbedding
        if isinstance(embedding_model, OpenAIEmbedding):
            openai_embedding_limit = 8191
            new_contents = openai_truncate_by_token(new_contents, openai_embedding_limit, embedding_model.model_name)

        new_ids = new_passage['doc_id'].tolist()
        embedded_contents = embedding_model.get_text_embedding_batch(new_contents, show_progress=True)
        input_batches = create_batches(api=collection._client, ids=new_ids, embeddings=embedded_contents)
        for batch in input_batches:
            ids = batch[0]
            embed_content = batch[1]
            collection.add(ids=ids, embeddings=embed_content)
