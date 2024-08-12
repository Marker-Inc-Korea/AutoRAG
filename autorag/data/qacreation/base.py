import logging
import uuid
from typing import Callable, Optional
import os
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
import faiss
import torch
import gc

from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from FlagEmbedding import BGEM3FlagModel
from autorag.utils.util import save_parquet_safe

logger = logging.getLogger("AutoRAG")


def make_single_content_qa(corpus_df: pd.DataFrame,
                           content_size: int,
                           qa_creation_func: Callable,
                           output_filepath: Optional[str] = None,
                           upsert: bool = False,
                           random_state: int = 42,
                           cache_batch: int = 32,
                           existing_query_df: Optional[pd.DataFrame] = None,
                           local: bool = False,
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
    :param existing_query_df: If given, the function will use the given existing_query_df to create QA pairs under FAISS.
    :param local: If true, the function will use local embedding model.
    :param kwargs: The keyword arguments for qa_creation_func.
    :return: QA dataset dataframe.
        You can save this as parquet file to use at AutoRAG.
    """

    assert content_size > 0, "content_size must be greater than 0."
    if content_size > len(corpus_df):
        logger.warning(f"content_size {content_size} is larger than the corpus size {len(corpus_df)}. "
                       "Setting content_size to the corpus size.")
        content_size = len(corpus_df)


    def get_embeddings(texts, local: bool = False):
        if local:
            logger.info("Loading local embedding model...")  
            model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device='cuda')
            return model.encode(texts, return_dense=True, max_length=512)
        else:
            logger.info("Loading OpenAI embedding model...")  
            model = OpenAIEmbeddings(model ="text-embedding-3-large")
            return model.embed_documents(texts, chunk_size=512)


        
    if existing_query_df is not None:
        if local:
            corpus_embed_file = 'corpus_embeddings.npy'
            query_embed_file = 'query_embeddings.npy'        
            
            if os.path.exists(corpus_embed_file) and os.path.exists(query_embed_file):
                logger.info("Loading existing local embeddings...")  
                corpus_embeddings_dict = np.load(corpus_embed_file, allow_pickle=True).item()
                corpus_embeddings = corpus_embeddings_dict['dense_vecs']
                query_embeddings_dict = np.load(query_embed_file, allow_pickle=True).item()
                query_embeddings = query_embeddings_dict['dense_vecs']
            else:
                logger.info("Calculating embeddings locally...")
                corpus_embeddings_dict = np.array(get_embeddings(corpus_df['contents'].tolist()), local).item()['dense_vecs']
                corpus_embeddings = corpus_embeddings_dict.item()['dense_vecs']
                query_embeddings_dict = np.array(get_embeddings(existing_query_df['query'].tolist()), local).item()['dense_vecs']
                query_embeddings = query_embeddings_dict.item()['dense_vecs']
                
                np.save(corpus_embed_file, corpus_embeddings_dict)
                np.save(query_embed_file, query_embeddings_dict)
        
        else:
            corpus_embed_file = 'corpus_embeddings_openAI.npy'
            query_embed_file = 'query_embeddings_openAI.npy'
            
            if os.path.exists(corpus_embed_file) and os.path.exists(query_embed_file):
                logger.info("Loading existing OpenAI embeddings...")  
                corpus_embeddings = np.array(np.load(corpus_embed_file, allow_pickle=True).tolist())
                query_embeddings = np.array(np.load(query_embed_file, allow_pickle=True).tolist())
            else:
                logger.info("Calculating embeddings with OpenAI...")
                corpus_embeddings = np.array(get_embeddings(corpus_df['contents'].tolist()))
                query_embeddings = np.array(get_embeddings(existing_query_df['query'].tolist()))
                    
                np.save(corpus_embed_file, corpus_embeddings)
                np.save(query_embed_file, query_embeddings)
                
        try:
            print(f"Corpus embeddings shape before indexing: {corpus_embeddings.shape}")
            if len(corpus_embeddings.shape) != 2:
                raise ValueError(f"Expected 2D array, got shape {corpus_embeddings.shape}")
            dimension = corpus_embeddings.shape[1]
            
            print("Creating CPU index...")
            index = faiss.IndexFlatL2(dimension)
            
            print("Adding embeddings to index...")
            index.add(corpus_embeddings.astype(np.float32)) 
            
            print("Clearing memory...")
            gc.collect()
            torch.cuda.empty_cache()        
            
            print("Performing search...")
            _, indices = index.search(query_embeddings.astype(np.float32), k=30)
            
            print("Getting relevant contents...")
            most_relevant_indices = indices[:, 0]
            relevant_contents = corpus_df.iloc[most_relevant_indices]            
            
            combined_df = relevant_contents.reset_index(drop=True)
            combined_df['query'] = existing_query_df['query'].values
            combined_df = combined_df.sample(n=content_size, random_state=random_state)
            
        except Exception as e:
            print(f"Embedding dimension: {corpus_embeddings.shape[1]}")
            logger.error(f"Error in FAISS indexing: {e}")

    else:
        sampled_corpus = corpus_df.sample(n=content_size, random_state=random_state)
        combined_df = sampled_corpus.reset_index(drop=True)


    def make_query_generation_gt(row):
        if existing_query_df is not None:
            return row['query'], row['qa']['generation_gt']
        else:
            return row['qa']['query'], row['qa']['generation_gt']


    qa_data = pd.DataFrame()
    print('total trial:', len(combined_df))
    for idx, i in tqdm(enumerate(range(0, len(combined_df), cache_batch)), total=len(combined_df)//cache_batch):
        print(f'Processing batch {idx+1}/{len(combined_df)//cache_batch + 1}')      
        batch = combined_df.iloc[i:i + cache_batch]
        
        if existing_query_df is not None:
            qa = qa_creation_func(contents=batch['contents'].tolist(), 
                                  queries=batch['query'].tolist(),
                                  **kwargs)
        else:
            qa = qa_creation_func(contents=batch['contents'].tolist(), **kwargs)

        temp_qa_data = pd.DataFrame({
            'qa': qa,
            'retrieval_gt': batch['doc_id'].tolist(),
        })
        
        if existing_query_df is not None:
            temp_qa_data['query'] = batch['query'].tolist()

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
