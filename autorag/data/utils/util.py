import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from langchain_core.documents import Document


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


def add_essential_metadata(metadata: Dict) -> Dict:
    if 'last_modified_datetime' not in metadata:
        metadata['last_modified_datetime'] = datetime.now()
    return metadata


def corpus_df_to_langchain_documents(corpus_df: pd.DataFrame) -> List[Document]:
    page_contents = corpus_df['contents'].tolist()
    ids = corpus_df['doc_id'].tolist()
    metadatas = corpus_df['metadata'].tolist()
    return list(map(lambda x: Document(page_content=x[0], metadata={'filename': x[1], **x[2]}),
                    zip(page_contents, ids, metadatas)))
