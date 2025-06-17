import logging
import uuid
from typing import Callable, Optional, List

import chromadb
import numpy as np
import pandas as pd
from tqdm import tqdm

import autorag
from autorag.nodes.retrieval.vectordb import vectordb_ingest, vectordb_pure
from autorag.utils.util import (
	save_parquet_safe,
	fetch_contents,
	get_event_loop,
	process_batch,
)

logger = logging.getLogger("AutoRAG")


def make_single_content_qa(
	corpus_df: pd.DataFrame,
	content_size: int,
	qa_creation_func: Callable,
	output_filepath: Optional[str] = None,
	upsert: bool = False,
	random_state: int = 42,
	cache_batch: int = 32,
	**kwargs,
) -> pd.DataFrame:
	"""
	Make single content (single-hop, single-document) QA dataset using given qa_creation_func.
	It generates a single content QA dataset, which means its retrieval ground truth will be only one.
	It is the most basic form of QA dataset.

	:param corpus_df: The corpus dataframe to make QA dataset from.
	:param content_size: This function will generate QA dataset for the given number of contents.
	:param qa_creation_func: The function to create QA pairs.
	    You can use like `generate_qa_llama_index` or `generate_qa_llama_index_by_ratio`.
	    The input func must have `contents` parameter for the list of content string.
	:param output_filepath: Optional filepath to save the parquet file.
	    If None, the function will return the processed_data as pd.DataFrame, but do not save as parquet.
	    File directory must exist. File extension must be .parquet
	:param upsert: If true, the function will overwrite the existing file if it exists.
	    Default is False.
	:param random_state: The random state for sampling corpus from the given corpus_df.
	:param cache_batch: The number of batches to use for caching the generated QA dataset.
	    When the cache_batch size data is generated, the dataset will save to the designated output_filepath.
	    If the cache_batch size is too small, the process time will be longer.
	:param kwargs: The keyword arguments for qa_creation_func.
	:return: QA dataset dataframe.
	    You can save this as parquet file to use at AutoRAG.
	"""
	assert content_size > 0, "content_size must be greater than 0."
	if content_size > len(corpus_df):
		logger.warning(
			f"content_size {content_size} is larger than the corpus size {len(corpus_df)}. "
			"Setting content_size to the corpus size."
		)
		content_size = len(corpus_df)
	sampled_corpus = corpus_df.sample(n=content_size, random_state=random_state)
	sampled_corpus = sampled_corpus.reset_index(drop=True)

	def make_query_generation_gt(row):
		return row["qa"]["query"], row["qa"]["generation_gt"]

	qa_data = pd.DataFrame()
	for idx, i in tqdm(enumerate(range(0, len(sampled_corpus), cache_batch))):
		qa = qa_creation_func(
			contents=sampled_corpus["contents"].tolist()[i : i + cache_batch], **kwargs
		)

		temp_qa_data = pd.DataFrame(
			{
				"qa": qa,
				"retrieval_gt": sampled_corpus["doc_id"].tolist()[i : i + cache_batch],
			}
		)
		temp_qa_data = temp_qa_data.explode("qa", ignore_index=True)
		temp_qa_data["qid"] = [str(uuid.uuid4()) for _ in range(len(temp_qa_data))]
		temp_qa_data[["query", "generation_gt"]] = temp_qa_data.apply(
			make_query_generation_gt, axis=1, result_type="expand"
		)
		temp_qa_data = temp_qa_data.drop(columns=["qa"])

		temp_qa_data["retrieval_gt"] = temp_qa_data["retrieval_gt"].apply(
			lambda x: [[x]]
		)
		temp_qa_data["generation_gt"] = temp_qa_data["generation_gt"].apply(
			lambda x: [x]
		)

		if idx == 0:
			qa_data = temp_qa_data
		else:
			qa_data = pd.concat([qa_data, temp_qa_data], ignore_index=True)
		if output_filepath is not None:
			save_parquet_safe(qa_data, output_filepath, upsert=upsert)

	return qa_data


def make_qa_with_existing_qa(
	corpus_df: pd.DataFrame,
	existing_query_df: pd.DataFrame,
	content_size: int,
	answer_creation_func: Optional[Callable] = None,
	exist_gen_gt: Optional[bool] = False,
	output_filepath: Optional[str] = None,
	embedding_model: str = "openai_embed_3_large",
	collection: Optional[chromadb.Collection] = None,
	upsert: bool = False,
	random_state: int = 42,
	cache_batch: int = 32,
	top_k: int = 3,
	**kwargs,
) -> pd.DataFrame:
	"""
	Make single-hop QA dataset using given qa_creation_func and existing queries.

	:param corpus_df: The corpus dataframe to make QA dataset from.
	:param existing_query_df: Dataframe containing existing queries to use for QA pair creation.
	:param content_size: This function will generate QA dataset for the given number of contents.
	:param answer_creation_func: Optional function to create answer with input query.
	    If exist_gen_gt is False, this function must be given.
	:param exist_gen_gt: Optional boolean to use existing generation_gt.
	    If True, the existing_query_df must have 'generation_gt' column.
	    If False, the answer_creation_func must be given.
	:param output_filepath: Optional filepath to save the parquet file.
	:param embedding_model: The embedding model to use for vectorization.
	    You can add your own embedding model in the autorag.embedding_models.
	    Please refer to how to add an embedding model in this doc: https://marker-inc-korea.github.io/AutoRAG/local_model.html
	    The default is 'openai_embed_3_large'.
	:param collection: The chromadb collection to use for vector DB.
	    You can make any chromadb collection and use it here.
	    If you already ingested the corpus_df to the collection, the embedding process will not be repeated.
	    The default is None. If None, it makes a temporary collection.
	:param upsert: If true, the function will overwrite the existing file if it exists.
	:param random_state: The random state for sampling corpus from the given corpus_df.
	:param cache_batch: The number of batches to use for caching the generated QA dataset.
	:param top_k: The number of sources to refer by model.
	    Default is 3.
	:param kwargs: The keyword arguments for qa_creation_func.
	:return: QA dataset dataframe.
	"""
	raise DeprecationWarning("This function is deprecated.")
	assert (
		"query" in existing_query_df.columns
	), "existing_query_df must have 'query' column."

	if exist_gen_gt:
		assert (
			"generation_gt" in existing_query_df.columns
		), "existing_query_df must have 'generation_gt' column."
	else:
		assert (
			answer_creation_func is not None
		), "answer_creation_func must be given when exist_gen_gt is False."

	assert content_size > 0, "content_size must be greater than 0."
	if content_size > len(corpus_df):
		logger.warning(
			f"content_size {content_size} is larger than the corpus size {len(corpus_df)}. "
			"Setting content_size to the corpus size."
		)
		content_size = len(corpus_df)

	logger.info("Loading local embedding model...")
	embeddings = autorag.embedding_models[embedding_model]()

	# Vector DB creation
	if collection is None:
		chroma_client = chromadb.Client()
		collection_name = "auto-rag"
		collection = chroma_client.get_or_create_collection(collection_name)

	# embed corpus_df
	vectordb_ingest(collection, corpus_df, embeddings)
	query_embeddings = embeddings.get_text_embedding_batch(
		existing_query_df["query"].tolist()
	)

	loop = get_event_loop()
	tasks = [
		vectordb_pure([query_embedding], top_k, collection)
		for query_embedding in query_embeddings
	]
	results = loop.run_until_complete(process_batch(tasks, batch_size=cache_batch))
	retrieved_ids = list(map(lambda x: x[0], results))

	retrieved_contents: List[List[str]] = fetch_contents(corpus_df, retrieved_ids)
	input_passage_strs: List[str] = list(
		map(
			lambda x: "\n".join(
				[f"Document {i + 1}\n{content}" for i, content in enumerate(x)]
			),
			retrieved_contents,
		)
	)

	retrieved_qa_df = pd.DataFrame(
		{
			"qid": [str(uuid.uuid4()) for _ in range(len(existing_query_df))],
			"query": existing_query_df["query"].tolist(),
			"retrieval_gt": list(map(lambda x: [x], retrieved_ids)),
			"input_passage_str": input_passage_strs,
		}
	)

	if exist_gen_gt:
		generation_gt = existing_query_df["generation_gt"].tolist()
		if isinstance(generation_gt[0], np.ndarray):
			retrieved_qa_df["generation_gt"] = generation_gt
		else:
			raise ValueError(
				"In existing_query_df, generation_gt (per query) must be in the form of List[str]."
			)

	sample_qa_df = retrieved_qa_df.sample(
		n=min(content_size, len(retrieved_qa_df)), random_state=random_state
	)

	qa_df = sample_qa_df.copy(deep=True)
	qa_df.drop(columns=["input_passage_str"], inplace=True)

	if not exist_gen_gt:
		generation_gt = answer_creation_func(
			contents=sample_qa_df["input_passage_str"].tolist(),
			queries=sample_qa_df["query"].tolist(),
			batch=cache_batch,
			**kwargs,
		)
		qa_df["generation_gt"] = generation_gt

	if output_filepath is not None:
		save_parquet_safe(qa_df, output_filepath, upsert=upsert)

	return qa_df
