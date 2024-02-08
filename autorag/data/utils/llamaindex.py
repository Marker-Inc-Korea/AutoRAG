import os
import pathlib
import mimetypes
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from llama_index.schema import Document as LlamaDocument


def get_file_metadata(file_path: str) -> Dict:
    """Get some handy metadate from filesystem.

    Args:
        file_path: str: file path in str
    """
    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_type": mimetypes.guess_type(file_path)[0],
        "file_size": os.path.getsize(file_path),
        "creation_datetime": datetime.fromtimestamp(
            Path(file_path).stat().st_ctime
        ).strftime("%Y-%m-%d"),
        "last_modified_datetime": datetime.fromtimestamp(
            Path(file_path).stat().st_mtime
        ).strftime("%Y-%m-%d"),
        "last_accessed_datetime": datetime.fromtimestamp(
            Path(file_path).stat().st_atime
        ).strftime("%Y-%m-%d"),
    }


def llama_documents_to_parquet(llama_documents: List[LlamaDocument],
                               output_filepath: str):
    """
    llama_documents to corpus_data
    corpus_data will be saved to filepath(file_dir/filename)

    :param llama_documents: List[LlamaDocument]
    :param output_filepath: file_dir must exist, filepath must not exist. file extension must be .parquet
    :return: corpus_data as pd.DataFrame
    """
    output_file_dir = pathlib.PurePath(output_filepath).parent
    if not os.path.isdir(output_file_dir):
        raise NotADirectoryError(f"directory {output_file_dir}  not found.")
    if not output_filepath.endswith("parquet"):
        raise NameError(f'file path: {output_filepath}  filename extension need to be ".parquet"')
    if os.path.exists(output_filepath):
        raise FileExistsError(f"{os.path.splitext(output_filepath)} already exists in {output_file_dir}.")

    doc_lst = list(map(lambda doc: {
        'doc_id': str(uuid.uuid4()),
        'contents': doc.text,
        'metadata': doc.metadata
    }, llama_documents))

    processed_data = pd.DataFrame(doc_lst)
    processed_data.to_parquet(output_filepath, index=False)

    return processed_data
