import uuid
from typing import List, Optional

import pandas as pd
from langchain_core.documents import Document

from autorag.data.utils.util import add_essential_metadata
from autorag.utils.util import save_parquet_safe


def langchain_documents_to_parquet(
	langchain_documents: List[Document],
	output_filepath: Optional[str] = None,
	upsert: bool = False,
) -> pd.DataFrame:
	"""
	Langchain documents to corpus dataframe.
	Corpus dataframe will be saved to filepath(file_dir/filename) if given.
	Return corpus dataframe whether the filepath is given.
	You can use this method to create corpus.parquet after load and chunk using Llama Index.

	:param langchain_documents: List of langchain documents.
	:param output_filepath: Optional filepath to save the parquet file.
	    If None, the function will return the processed_data as pd.DataFrame, but do not save as parquet.
	    File directory must exist. File extension must be .parquet
	:param upsert: If true, the function will overwrite the existing file if it exists.
	    Default is False.
	:return: Corpus data as pd.DataFrame
	"""

	corpus_df = pd.DataFrame(
		list(
			map(
				lambda doc: {
					"doc_id": str(uuid.uuid4()),
					"contents": doc.page_content,
					"metadata": add_essential_metadata(doc.metadata),
				},
				langchain_documents,
			)
		)
	)

	if output_filepath is not None:
		save_parquet_safe(corpus_df, output_filepath, upsert=upsert)

	return corpus_df
