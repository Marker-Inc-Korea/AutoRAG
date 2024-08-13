import logging
import uuid
from typing import Callable, Optional, List

import pandas as pd
from tqdm import tqdm

import os
import chromadb
from langchain.vectorstores import Chroma
from chromadb.utils import embedding_functions
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from FlagEmbedding import BGEM3FlagModel

from autorag.utils.util import save_parquet_safe

logger = logging.getLogger("AutoRAG")




class BGEM3Embeddings(Embeddings):
    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device='cuda')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, return_dense=True, max_length=512)
        return embeddings['dense_vecs'].tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], return_dense=True, max_length=512)
        return embedding['dense_vecs'][0].tolist()




def make_single_content_qa(corpus_df: pd.DataFrame,
                           content_size: int,
                           qa_creation_func: Callable,
                           output_filepath: Optional[str] = None,
                           upsert: bool = False,
                           random_state: int = 42,
                           cache_batch: int = 32,
                           **kwargs) -> pd.DataFrame:
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
        logger.warning(f"content_size {content_size} is larger than the corpus size {len(corpus_df)}. "
                       "Setting content_size to the corpus size.")
        content_size = len(corpus_df)
    sampled_corpus = corpus_df.sample(n=content_size, random_state=random_state)
    sampled_corpus = sampled_corpus.reset_index(drop=True)

    def make_query_generation_gt(row):
        return row['qa']['query'], row['qa']['generation_gt']

    qa_data = pd.DataFrame()
    for idx, i in tqdm(enumerate(range(0, len(sampled_corpus), cache_batch))):
        qa = qa_creation_func(contents=sampled_corpus['contents'].tolist()[i:i + cache_batch], **kwargs)

        temp_qa_data = pd.DataFrame({
            'qa': qa,
            'retrieval_gt': sampled_corpus['doc_id'].tolist()[i:i + cache_batch],
        })
        temp_qa_data = temp_qa_data.explode('qa', ignore_index=True)
        temp_qa_data['qid'] = [str(uuid.uuid4()) for _ in range(len(temp_qa_data))]
        temp_qa_data[['query', 'generation_gt']] = temp_qa_data.apply(make_query_generation_gt, axis=1,
                                                                      result_type='expand')
        temp_qa_data = temp_qa_data.drop(columns=['qa'])

        temp_qa_data['retrieval_gt'] = temp_qa_data['retrieval_gt'].apply(lambda x: [[x]])
        temp_qa_data['generation_gt'] = temp_qa_data['generation_gt'].apply(lambda x: [x])

        if idx == 0:
            qa_data = temp_qa_data
        else:
            qa_data = pd.concat([qa_data, temp_qa_data], ignore_index=True)
        if output_filepath is not None:
            save_parquet_safe(qa_data, output_filepath, upsert=upsert)

    return qa_data




def make_qa_with_existing_queries(
    corpus_df: pd.DataFrame,
    existing_query_df: pd.DataFrame,
    content_size: int,
    qa_creation_func: Callable,
    output_filepath: Optional[str] = None,
    upsert: bool = False,
    random_state: int = 42,
    cache_batch: int = 32,
    local: bool = False,
    top_k: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Make single content QA dataset using given qa_creation_func and existing queries.
    
    :param corpus_df: The corpus dataframe to make QA dataset from.
    :param existing_query_df: Dataframe containing existing queries to use for QA pair creation.
    :param content_size: This function will generate QA dataset for the given number of contents.
    :param qa_creation_func: The function to create QA pairs.
    :param output_filepath: Optional filepath to save the parquet file.
    :param upsert: If true, the function will overwrite the existing file if it exists.
    :param random_state: The random state for sampling corpus from the given corpus_df.
    :param cache_batch: The number of batches to use for caching the generated QA dataset.
    :param local: If true, the function will use local embedding model.
    :param top_k: The number of sources to refer by model.
    :param kwargs: The keyword arguments for qa_creation_func.
    :return: QA dataset dataframe.
    """
    assert content_size > 0, "content_size must be greater than 0."
    if content_size > len(corpus_df):
        logger.warning(f"content_size {content_size} is larger than the corpus size {len(corpus_df)}. "
                       "Setting content_size to the corpus size.")
        content_size = len(corpus_df)

    if local:
        logger.info("Loading local embedding model...")
        embeddings = BGEM3Embeddings()
        persist_directory = './chroma_db_local'
    else:
        logger.info("Loading OpenAI embedding model...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        persist_directory = './chroma_db_OpenAI'

    # Vector DB creation or loading
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        logger.info("Loaded existing vector database.")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_texts(
            texts=corpus_df['contents'].tolist(),
            embedding=embeddings,
            metadatas=[{"doc_id": doc_id} for doc_id in corpus_df['doc_id']],
            persist_directory=persist_directory
        )
        logger.info("Created and saved new vector database.")

    results = []
    for query in existing_query_df['query'].tolist():
        result = vectorstore.similarity_search(query, k=top_k)
        results.append(result)

    relevant_docs = [doc for batch in results for doc in batch]
    relevant_doc_ids = [[doc.metadata['doc_id'] for doc in batch] for batch in results]
    
    combined_df = pd.DataFrame({
        'doc_id': [doc_id for batch in relevant_doc_ids for doc_id in batch],
        'query': [query for query, batch in zip(existing_query_df['query'], relevant_doc_ids) for _ in batch]
    })
    
    content_series = corpus_df.set_index('doc_id')['contents']
    combined_df['contents'] = combined_df['doc_id'].map(content_series)
    combined_df['contents'] = combined_df['contents'].fillna('')
    
    combined_df = combined_df.sample(n=min(content_size, len(combined_df)), random_state=random_state)

    qa_data = pd.DataFrame()
    for idx, i in tqdm(enumerate(range(0, len(combined_df), cache_batch)), total=len(combined_df)//cache_batch):
        batch = combined_df.iloc[i:i + cache_batch]
        
        qa = qa_creation_func(contents=batch['contents'].tolist(), 
                              queries=batch['query'].tolist(),
                              **kwargs)

        temp_qa_data = pd.DataFrame({
            'qa': qa,
            'retrieval_gt': batch['doc_id'].tolist(),
            'query': batch['query'].tolist()
        })

        temp_qa_data = temp_qa_data.explode('qa', ignore_index=True)
        temp_qa_data['qid'] = [str(uuid.uuid4()) for _ in range(len(temp_qa_data))]
        temp_qa_data[['query', 'generation_gt']] = temp_qa_data.apply(
            lambda row: (row['query'], row['qa']['generation_gt']), axis=1, result_type='expand')
        temp_qa_data = temp_qa_data.drop(columns=['qa'])

        temp_qa_data['retrieval_gt'] = temp_qa_data['retrieval_gt'].apply(lambda x: [[x]])
        temp_qa_data['generation_gt'] = temp_qa_data['generation_gt'].apply(lambda x: [x])

        if idx == 0:
            qa_data = temp_qa_data
        else:
            qa_data = pd.concat([qa_data, temp_qa_data], ignore_index=True)
        
        if output_filepath is not None:
            save_parquet_safe(qa_data, output_filepath, upsert=upsert)
            
    return qa_data