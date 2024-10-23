import itertools
import logging
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from autorag.evaluation.metric.util import (
	calculate_l2_distance,
	calculate_inner_product,
	calculate_cosine_similarity,
)
from autorag.nodes.retrieval.base import evenly_distribute_passages, BaseRetrieval
from autorag.utils import (
	validate_corpus_dataset,
	cast_corpus_dataset,
	cast_qa_dataset,
	validate_qa_dataset,
)
from autorag.utils.util import (
	get_event_loop,
	process_batch,
	openai_truncate_by_token,
	flatten_apply,
	result_to_dataframe,
	pop_params,
	fetch_contents,
	empty_cuda_cache,
	convert_inputs_to_list,
	make_batch,
)
from autorag.vectordb import load_vectordb_from_yaml
from autorag.vectordb.base import BaseVectorStore

logger = logging.getLogger("AutoRAG")


class VectorDB(BaseRetrieval):
	def __init__(self, project_dir: str, vectordb: str = "default", **kwargs):
		"""
		Initialize VectorDB retrieval node.

		:param project_dir: The project directory path.
		:param vectordb: The vectordb name.
			You must configure the vectordb name in the config.yaml file.
			If you don't configure, it uses the default vectordb.
		:param kwargs: The optional arguments.
			Not affected in the init method.
		"""
		super().__init__(project_dir)

		vectordb_config_path = os.path.join(self.resources_dir, "vectordb.yaml")
		self.vector_store = load_vectordb_from_yaml(
			vectordb_config_path, vectordb, project_dir
		)

		self.embedding_model = self.vector_store.embedding

	def __del__(self):
		del self.vector_store
		del self.embedding_model
		empty_cuda_cache()
		super().__del__()

	@result_to_dataframe(["retrieved_contents", "retrieved_ids", "retrieve_scores"])
	def pure(self, previous_result: pd.DataFrame, *args, **kwargs):
		queries = self.cast_to_run(previous_result)
		pure_params = pop_params(self._pure, kwargs)
		ids, scores = self._pure(queries, **pure_params)
		contents = fetch_contents(self.corpus_df, ids)
		return contents, ids, scores

	def _pure(
		self,
		queries: List[List[str]],
		top_k: int,
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
		:param embedding_batch: The number of queries to be processed in parallel.
		    This is used to prevent API error at the query embedding.
		    Default is 128.
		:param ids: The optional list of ids that you want to retrieve.
		    You don't need to specify this in the general use cases.
		    Default is None.

		:return: The 2-d list contains a list of passage ids that retrieved from vectordb and 2-d list of its scores.
		    It will be a length of queries. And each element has a length of top_k.
		"""
		# if ids are specified, fetch the ids score from Chroma
		if ids is not None:
			return self.__get_ids_scores(queries, ids, embedding_batch)

		# run async vector_db_pure function
		tasks = [
			vectordb_pure(query_list, top_k, self.vector_store)
			for query_list in queries
		]
		loop = get_event_loop()
		results = loop.run_until_complete(
			process_batch(tasks, batch_size=embedding_batch)
		)
		id_result = list(map(lambda x: x[0], results))
		score_result = list(map(lambda x: x[1], results))
		return id_result, score_result

	def __get_ids_scores(self, queries, ids, embedding_batch: int):
		# truncate queries and embedding execution here.
		openai_embedding_limit = 8000
		if isinstance(self.embedding_model, OpenAIEmbedding):
			queries = list(
				map(
					lambda query_list: openai_truncate_by_token(
						query_list,
						openai_embedding_limit,
						self.embedding_model.model_name,
					),
					queries,
				)
			)

		query_embeddings = flatten_apply(
			run_query_embedding_batch,
			queries,
			embedding_model=self.embedding_model,
			batch_size=embedding_batch,
		)

		loop = get_event_loop()

		async def run_fetch(ids):
			final_result = []
			for id_list in ids:
				if len(id_list) == 0:
					final_result.append([])
				else:
					result = await self.vector_store.fetch(id_list)
					final_result.append(result)
			return final_result

		content_embeddings = loop.run_until_complete(run_fetch(ids))

		score_result = list(
			map(
				lambda query_embedding_list, content_embedding_list: get_id_scores(
					query_embedding_list,
					content_embedding_list,
					similarity_metric=self.vector_store.similarity_metric,
				),
				query_embeddings,
				content_embeddings,
			)
		)
		return ids, score_result


async def vectordb_pure(
	queries: List[str], top_k: int, vectordb: BaseVectorStore
) -> Tuple[List[str], List[float]]:
	"""
	Async VectorDB retrieval function.
	Its usage is for async retrieval of vector_db row by row.

	:param query_embeddings: A list of query embeddings.
	:param top_k: The number of passages to be retrieved.
	:param vectordb: The vector store instance.
	:return: The tuple contains a list of passage ids that are retrieved from vectordb and a list of its scores.
	"""
	id_result, score_result = await vectordb.query(queries=queries, top_k=top_k)

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


async def filter_exist_ids(
	vectordb: BaseVectorStore,
	corpus_data: pd.DataFrame,
) -> pd.DataFrame:
	corpus_data = cast_corpus_dataset(corpus_data)
	validate_corpus_dataset(corpus_data)
	ids = corpus_data["doc_id"].tolist()

	# Query the collection to check if IDs already exist
	existed_bool_list = await vectordb.is_exist(ids=ids)
	# Assuming 'ids' is the key in the response
	new_passage = corpus_data[~pd.Series(existed_bool_list)]
	return new_passage


async def filter_exist_ids_from_retrieval_gt(
	vectordb: BaseVectorStore,
	qa_data: pd.DataFrame,
	corpus_data: pd.DataFrame,
) -> pd.DataFrame:
	qa_data = cast_qa_dataset(qa_data)
	validate_qa_dataset(qa_data)
	corpus_data = cast_corpus_dataset(corpus_data)
	validate_corpus_dataset(corpus_data)
	retrieval_gt = (
		qa_data["retrieval_gt"]
		.apply(lambda x: list(itertools.chain.from_iterable(x)))
		.tolist()
	)
	retrieval_gt = list(itertools.chain.from_iterable(retrieval_gt))
	retrieval_gt = list(set(retrieval_gt))

	existed_bool_list = await vectordb.is_exist(ids=retrieval_gt)
	add_ids = []
	for ret_gt, is_exist in zip(retrieval_gt, existed_bool_list):
		if not is_exist:
			add_ids.append(ret_gt)
	new_passage = corpus_data[corpus_data["doc_id"].isin(add_ids)]
	return new_passage


async def vectordb_ingest(
	vectordb: BaseVectorStore,
	corpus_data: pd.DataFrame,
):
	"""
	Ingest given corpus data to the vectordb.
	It truncates corpus content when the embedding model is OpenAIEmbedding to the 8000 tokens.
	Plus, when the corpus content is empty (whitespace), it will be ignored.
	And if there is a document id that already exists in the collection, it will be ignored.

	:param vectordb: A vector stores instance that you want to ingest.
	:param corpus_data: The corpus data that contains doc_id and contents columns.
	"""
	embedding_batch = vectordb.embedding_batch
	if not corpus_data.empty:
		new_contents = corpus_data["contents"].tolist()
		new_ids = corpus_data["doc_id"].tolist()
		content_batches = make_batch(new_contents, embedding_batch)
		id_batches = make_batch(new_ids, embedding_batch)
		for content_batch, id_batch in zip(content_batches, id_batches):
			await vectordb.add(ids=id_batch, texts=content_batch)


def run_query_embedding_batch(
	queries: List[str], embedding_model: BaseEmbedding, batch_size: int
) -> List[List[float]]:
	result = []
	for i in range(0, len(queries), batch_size):
		batch = queries[i : i + batch_size]
		embeddings = embedding_model.get_text_embedding_batch(batch)
		result.extend(embeddings)
	return result


@convert_inputs_to_list
def get_id_scores(  # To find the uncalculated score when fuse the scores for the hybrid retrieval
	query_embeddings: List[
		List[float]
	],  # `queries` is input. This is one user input query.
	content_embeddings: List[List[float]],
	similarity_metric: str,
) -> List[
	float
]:  # The most high scores among each query. The length of a result is the same as the contents length.
	"""
	Calculate the highest similarity scores between query embeddings and content embeddings.

	:param query_embeddings: A list of lists containing query embeddings.
	:param content_embeddings: A list of lists containing content embeddings.
	:param similarity_metric: The similarity metric to use ('l2', 'ip', or 'cosine').
	:return: A list of the highest similarity scores for each content embedding.
	"""
	metric_func_dict = {
		"l2": lambda x, y: 1 - calculate_l2_distance(x, y),
		"ip": calculate_inner_product,
		"cosine": calculate_cosine_similarity,
	}
	metric_func = metric_func_dict[similarity_metric]

	result = []
	for content_embedding in content_embeddings:
		scores = []
		for query_embedding in query_embeddings:
			scores.append(
				metric_func(np.array(query_embedding), np.array(content_embedding))
			)
		result.append(max(scores))
	return result
