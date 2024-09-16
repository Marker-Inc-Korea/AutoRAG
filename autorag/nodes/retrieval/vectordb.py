from typing import List, Tuple, Optional

import chromadb
import pandas as pd
from chromadb import GetResult, QueryResult
from chromadb.utils.batch_utils import create_batches
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.nodes.retrieval.base import retrieval_node, evenly_distribute_passages
from autorag.utils import validate_corpus_dataset, cast_corpus_dataset
from autorag.utils.util import (
	get_event_loop,
	process_batch,
	openai_truncate_by_token,
	flatten_apply,
)


@retrieval_node
def vectordb(
	queries: List[List[str]],
	top_k: int,
	collection: chromadb.Collection,
	embedding_model: BaseEmbedding,
	embedding_batch: int = 128,
	ids: Optional[List[List[str]]] = None,
) -> Tuple[List[List[str]], List[List[float]]]:
	"""
	VectorDB retrieval function.
	You have to get a chroma collection that is already ingested.
	You have to get an embedding model that is already used in ingesting.

	:param queries: 2-d list of query strings.
	    Each element of the list is a query strings of each row.
	:param top_k: The number of passages to be retrieved.
	:param collection: A chroma collection instance that will be used to retrieve passages.
	:param embedding_model: An embedding model instance that will be used to embed queries.
	:param embedding_batch: The number of queries to be processed in parallel.
	    This is used to prevent API error at the query embedding.
	    Default is 128.
	:param ids: The optional list of ids that you want to retrieve.
	    You don't need to specify this in the general use cases.
	    Default is None.

	:return: The 2-d list contains a list of passage ids that retrieved from vectordb and 2-d list of its scores.
	    It will be a length of queries. And each element has a length of top_k.
	"""
	# check if bm25_corpus is valid
	assert (
		collection.count() > 0
	), "collection must contain at least one document. Please check you ingested collection correctly."
	# truncate queries and embedding execution here.
	openai_embedding_limit = 8191
	if isinstance(embedding_model, OpenAIEmbedding):
		queries = list(
			map(
				lambda query_list: openai_truncate_by_token(
					query_list, openai_embedding_limit, embedding_model.model_name
				),
				queries,
			)
		)

	query_embeddings = flatten_apply(
		run_query_embedding_batch,
		queries,
		embedding_model=embedding_model,
		batch_size=embedding_batch,
	)

	# if ids are specified, fetch the ids score from Chroma
	if ids is not None:
		client = chromadb.Client()
		score_result = list(
			map(
				lambda query_embedding_list, id_list: get_id_scores(
					id_list, query_embedding_list, collection, client
				),
				query_embeddings,
				ids,
			)
		)
		return ids, score_result

	# run async vector_db_pure function
	tasks = [
		vectordb_pure(query_embedding, top_k, collection)
		for query_embedding in query_embeddings
	]
	loop = get_event_loop()
	results = loop.run_until_complete(process_batch(tasks, batch_size=embedding_batch))
	id_result = list(map(lambda x: x[0], results))
	score_result = list(map(lambda x: x[1], results))
	return id_result, score_result


async def vectordb_pure(
	query_embeddings: List[List[float]], top_k: int, collection: chromadb.Collection
) -> Tuple[List[str], List[float]]:
	"""
	Async VectorDB retrieval function.
	Its usage is for async retrieval of vector_db row by row.

	:param query_embeddings: A list of query embeddings.
	:param top_k: The number of passages to be retrieved.
	:param collection: A chroma collection instance that will be used to retrieve passages.
	:return: The tuple contains a list of passage ids that are retrieved from vectordb and a list of its scores.
	"""
	id_result, score_result = [], []
	for embedded_query in query_embeddings:
		result = collection.query(query_embeddings=embedded_query, n_results=top_k)
		id_result.extend(result["ids"])
		score_result.extend(
			list(map(lambda lst: list(map(lambda x: 1 - x, lst)), result["distances"]))
		)

	# Distribute passages evenly
	id_result, score_result = evenly_distribute_passages(id_result, score_result, top_k)
	# sort id_result and score_result by score
	result = [
		(_id, score)
		for score, _id in sorted(
			zip(score_result, id_result), key=lambda pair: pair[0], reverse=True
		)
	]
	id_result, score_result = zip(*result)
	return list(id_result), list(score_result)


def vectordb_ingest(
	collection: chromadb.Collection,
	corpus_data: pd.DataFrame,
	embedding_model: BaseEmbedding,
	embedding_batch: int = 128,
):
	"""
	Ingest given corpus data to the chromadb collection.
	It truncates corpus content when the embedding model is OpenAIEmbedding to the 8191 tokens.
	Plus, when the corpus content is empty (whitespace), it will be ignored.
	And if there is a document id that already exists in the collection, it will be ignored.

	:param collection: Chromadb collection instance to ingest.
	:param corpus_data: The corpus data that contains doc_id and contents columns.
	:param embedding_model: An embedding model instance that will be used to embed queries.
	:param embedding_batch: The number of chunks that will be processed in parallel.
	"""
	embedding_model.embed_batch_size = embedding_batch
	corpus_data = cast_corpus_dataset(corpus_data)
	validate_corpus_dataset(corpus_data)
	ids = corpus_data["doc_id"].tolist()

	# Query the collection to check if IDs already exist
	existing_ids = set(
		collection.get(ids=ids)["ids"]
	)  # Assuming 'ids' is the key in the response
	new_passage = corpus_data[~corpus_data["doc_id"].isin(existing_ids)]

	if not new_passage.empty:
		new_contents = new_passage["contents"].tolist()

		# truncate by token if embedding_model is OpenAIEmbedding
		if isinstance(embedding_model, OpenAIEmbedding):
			openai_embedding_limit = (
				8191  # all openai embedding models have 8191 max token input
			)
			new_contents = openai_truncate_by_token(
				new_contents, openai_embedding_limit, embedding_model.model_name
			)

		new_ids = new_passage["doc_id"].tolist()
		embedded_contents = embedding_model.get_text_embedding_batch(
			new_contents, show_progress=True
		)
		input_batches = create_batches(
			api=collection._client, ids=new_ids, embeddings=embedded_contents
		)
		for batch in input_batches:
			ids = batch[0]
			embed_content = batch[1]
			collection.add(ids=ids, embeddings=embed_content)


def run_query_embedding_batch(
	queries: List[str], embedding_model: BaseEmbedding, batch_size: int
) -> List[List[float]]:
	result = []
	for i in range(0, len(queries), batch_size):
		batch = queries[i : i + batch_size]
		embeddings = embedding_model.get_text_embedding_batch(batch)
		result.extend(embeddings)
	return result


def get_id_scores(
	ids: List[str],
	query_embeddings: List[List[float]],
	collection: chromadb.Collection,
	temp_client: chromadb.Client,
) -> List[float]:
	if len(ids) == 0 or ids is None or not bool(ids):
		return []

	id_results: GetResult = collection.get(ids, include=["embeddings"])
	temp_collection = temp_client.create_collection(
		name="temp", metadata={"hnsw:space": "cosine"}
	)
	temp_collection.add(ids=id_results["ids"], embeddings=id_results["embeddings"])

	query_result: QueryResult = temp_collection.query(
		query_embeddings=query_embeddings, n_results=len(ids)
	)
	assert len(query_result["ids"]) == len(query_result["distances"])
	id_scores_dict = {id_: [] for id_ in ids}
	for id_list, score_list in zip(query_result["ids"], query_result["distances"]):
		for id_ in list(id_scores_dict.keys()):
			id_idx = id_list.index(id_)
			id_scores_dict[id_].append(score_list[id_idx])
	id_scores_pd = pd.DataFrame(id_scores_dict)
	temp_client.delete_collection("temp")
	return id_scores_pd.max(axis=0).tolist()
