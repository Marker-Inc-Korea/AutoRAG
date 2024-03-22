import uuid
from typing import List, Optional

import pandas as pd
from llama_index.core import Document

from autorag.utils.util import save_parquet_safe


def llama_documents_to_parquet(llama_documents: List[Document],
                               output_filepath: Optional[str] = None,
                               upsert: bool = False) -> pd.DataFrame:
    """
    Llama documents to corpus dataframe.
    Corpus dataframe will be saved to filepath(file_dir/filename) if given.
    Return corpus dataframe whether the filepath is given.
    You can use this method to create corpus.parquet after load and chunk using Llama Index.

    :param llama_documents: List[Document]
    :param output_filepath: Optional filepath to save the parquet file.
        If None, the function will return the processed_data as pd.DataFrame, but do not save as parquet.
        File directory must exist. File extension must be .parquet
    :param upsert: If true, the function will overwrite the existing file if it exists.
        Default is False.
    :return: Corpus data as pd.DataFrame
    """
    doc_lst = list(map(lambda doc: {
        'doc_id': str(uuid.uuid4()),
        'contents': doc.text,
        'metadata': doc.metadata
    }, llama_documents))
    processed_df = pd.DataFrame(doc_lst)

    if output_filepath is not None:
        save_parquet_safe(processed_df, output_filepath, upsert=upsert)

    return processed_df
