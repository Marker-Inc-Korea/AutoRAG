---
myst:
   html_meta:
      title: AutoRAG - creating your own RAG evaluation dataset
      description: Check out AutoRAG dataset format. You will know about how to make AutoRAG compatible RAG evaluation dataset.
      keywords: AutoRAG,RAG,RAG evaluation,RAG dataset
---
# Start creating your own evaluation data

## Index

1. [Overview](#overview)
2. [Raw data to Corpus data](#make-corpus-data-from-raw-documents)
3. [Corpus data to QA data](#make-qa-data-from-corpus-data)
4. [Use custom prompt](#use-custom-prompt)
5. [Use multiple prompts](#use-multiple-prompts)
6. [If there are existing queries](#when-you-have-existing-qa-data)


## Overview
For the evaluation of RAGs we need data, but in most cases we have little or no satisfactory data.

However, since the advent of LLM, creating synthetic data has become one of the good solutions to this problem.

The following guide covers how to use LLM to create data in a form that AutoRAG can use.

---
![Data Creation](../_static/data_creation.png)

AutoRAG aims to work with Python’s ‘primitive data types’ for scalability and convenience.

Therefore, to use AutoRAG, you need to convert your raw data into `corpus data`  and `qa data` to our [data format](./data_format.md).


## Make corpus data from raw documents
1. Load your raw data to texts with loaders such as lama_index, LangChain, etc.
2. Chunk the texts into passages. Use Langchain, LlamaIndex, etc.
3. Make it into corpus data to use converter functions.
   There are converter functions for llama index `Document`, `TextNode`, and Langchain `Document` objects,
   which is `llama_document_to_parquet`, `llama_text_node_to_parquet`, and `langchain_document_to_parquet`.

- Use Llama Index

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from autorag.data.legacy.corpus import llama_text_node_to_parquet

documents = SimpleDirectoryReader('your_dir_path').load_data()
nodes = TokenTextSplitter(chunk_size=512, chunk_overlap=128).get_nodes_from_documents(documents=documents)
corpus_df = llama_text_node_to_parquet(nodes, 'path/to/corpus.parquet')
```

- Use LangChain

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from autorag.data.legacy.corpus import langchain_documents_to_parquet

documents = DirectoryLoader('your_dir_path', glob='**/*.md').load()
documents = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128).split_documents(documents)
corpus_df = langchain_documents_to_parquet(documents, 'path/to/corpus.parquet')
```

```{tip}
The format for corpus data can be found [corpus data format](data_format.md#corpus-dataset)
```

## Make qa data from corpus data

```{tip}
The format for qa data can be found [qa data format](data_format.md#qa-dataset)
```

```python
import pandas as pd
from llama_index.llms.openai import OpenAI
from autorag.data.legacy.qacreation import generate_qa_llama_index, make_single_content_qa

corpus_df = pd.read_parquet('path/to/corpus.parquet')
llm = OpenAI(model='gpt-3.5-turbo', temperature=1.0)
qa_df = make_single_content_qa(corpus_df, 50, generate_qa_llama_index, llm=llm, question_num_per_content=1,
                               output_filepath='path/to/qa.parquet', cache_batch=64)
```

`generate_qa_llama_index` is a function designed to generate **questions** and its **generation_gt** per content.
You can set the number of questions per content by changing `question_num_per_content` parameter.

And the `make_single_content_qa` function is designed to generate `qa.parquet` file using input function.
It generates 'single content' qa data, also known as 'single-hop' or 'single-document' QA data.
Which means it uses only one passage per question for answering the question.

```{admonition} What is passage?
Passage is chunked units from raw data.
```

```{admonition} Auto-save feature
From AutoRAG v0.2.9, the auto-save feature added!
Now, you don't have to afraid that something wrong while the data generation.
The data will save automatically to the input `output_filepath`.

You can set how often you want to save the result to the file.
Just adjust `cache_batch` parameter. Default is 32.
```

## Use custom prompt

You can use custom prompt to generate qa data.
The prompt must contains two placeholders:

- {{text}}: The content string
- {{num_questions}}: The number of questions to generate

```python
import pandas as pd

from llama_index.llms.openai import OpenAI
from autorag.data.legacy.qacreation import generate_qa_llama_index, make_single_content_qa

prompt = """
Generate question and answer pairs for the given passage.

Passage:
{{text}}

Number of questions to generate: {{num_questions}}

Example:
[Q]: What is this?
[A]: This is a sample question.

Result:
"""

corpus_df = pd.read_parquet('path/to/corpus.parquet')
llm = OpenAI(model='gpt-3.5-turbo', temperature=1.0)
qa_df = make_single_content_qa(corpus_df, content_size=50, qa_creation_func=generate_qa_llama_index,
                               llm=llm, prompt=prompt, question_num_per_content=1)
```

## Use multiple prompts

If you want to generate different types of question and answer pairs, you can use multiple prompts.
From now, we support distributing multiple prompts by randomly based on the ratio of each prompt.
It means that the prompt will be selected by a ratio per passage.

For this, you must provide a dictionary.
The dictionary must have the key, which is the prompt text file path, and the value which is the ratio of the prompt.

```python
import pandas as pd
from llama_index.llms.openai import OpenAI
from autorag.data.legacy.qacreation import generate_qa_llama_index_by_ratio, make_single_content_qa

ratio_dict = {
    'prompt1.txt': 1,
    'prompt2.txt': 2,
    'prompt3.txt': 3
}

corpus_df = pd.read_parquet('path/to/corpus.parquet')
llm = OpenAI(model='gpt-3.5-turbo', temperature=1.0)
qa_df = make_single_content_qa(corpus_df, content_size=50, qa_creation_func=generate_qa_llama_index_by_ratio,
                               llm=llm, prompts_ratio=ratio_dict, question_num_per_content=1, batch=6)
```

```{warning}
Remeber all prompts must have the placeholders `{{text}}` and `{{num_questions}}`.
```

## When you have existing qa data

When you have existing qa data, you can use it for AutoRAG.
The real user's qa data is valuable data, so it is always great to use it prior to generating synthetic data.

But you have to make retrieval_gt for existing queries from your corpus data.
The process to find the retrieval_gt at the corpus is hard but must be accurate.
To make it less hard, we use an embedding model and vectordb for finding relevant passages.
After that, you have to clarify the retrieval_gt is right.
If retrieval_gt is not relevant, you have to remove it from the dataset.

This feature is available if you have only query ready, and if you have both query and generation_gt ready.

### If you only have query data:

First get retrieval_gt with the existing query, then put a query and retrieval_gt into LLM and generate generation_gt.

- `answer_creation_func`, `llm` parameters are necessary.
- `existing_qa_df` must have 'query' column.

```python
import pandas as pd
from llama_index.llms.openai import OpenAI
from autorag.data.legacy.qacreation import make_qa_with_existing_qa, generate_answers

corpus_df = pd.read_parquet('path/to/corpus.parquet')
existing_qa_df = pd.read_parquet('path/to/existing_qa.parquet')  # It has to contain 'query' column
llm = OpenAI(model='gpt-3.5-turbo', temperature=1.0)
qa_df = make_qa_with_existing_qa(corpus_df, existing_qa_df, content_size=50,
                                 answer_creation_func=generate_answers,
                                 llm=llm, output_filepath='path/to/qa.parquet', cache_batch=64,
                                 embedding_model='openai_embed_3_large', top_k=5)
```

You can use `PersistentClient` for saving corpus embeddings locally as well.

```python
import pandas as pd
import chromadb
from llama_index.llms.openai import OpenAI
from autorag.data.legacy.qacreation import make_qa_with_existing_qa, generate_answers

client = chromadb.PersistentClient('path/to/chromadb')
collection = client.get_or_create_collection('auto-rag')

corpus_df = pd.read_parquet('path/to/corpus.parquet')
existing_qa_df = pd.read_parquet('path/to/existing_qa.parquet')  # It has to contain 'query' column
llm = OpenAI(model='gpt-3.5-turbo', temperature=1.0)
qa_df = make_qa_with_existing_qa(corpus_df, existing_qa_df, content_size=50,
                                 answer_creation_func=generate_answers, collection=collection,
                                 llm=llm, output_filepath='path/to/qa.parquet', cache_batch=64,
                                 embedding_model='openai_embed_3_large', top_k=5)
```

### If you have both query and generation_gt:

Use a query and generation_gt as they are, and just find and add retrieval_gt.

- `answer_creation_func`, `llm` parameters are not necessary.
- `exist_gen_gt=True` parameter is necessary.
- `existing_qa_df` must have 'query' and 'generation_gt' columns.
   - generation_gt(per query) must be in the form of List[str].

```python
import pandas as pd
from llama_index.llms.openai import OpenAI
from autorag.data.legacy.qacreation import make_qa_with_existing_qa

corpus_df = pd.read_parquet('path/to/corpus.parquet')
existing_qa_df = pd.read_parquet(
    'path/to/existing_qa.parquet')  # It has to contain 'query' and 'generation_gt' columns.
llm = OpenAI(model='gpt-3.5-turbo', temperature=1.0)
qa_df = make_qa_with_existing_qa(corpus_df, existing_qa_df, content_size=50, exist_gen_gt=True,
                                 output_filepath='path/to/qa.parquet', cache_batch=64,
                                 embedding_model='openai_embed_3_large', top_k=5)
```

You can use `PersistentClient` for saving corpus embeddings locally as well.

```python
import pandas as pd
import chromadb
from llama_index.llms.openai import OpenAI
from autorag.data.legacy.qacreation import make_qa_with_existing_qa

client = chromadb.PersistentClient('path/to/chromadb')
collection = client.get_or_create_collection('auto-rag')

corpus_df = pd.read_parquet('path/to/corpus.parquet')
existing_qa_df = pd.read_parquet(
    'path/to/existing_qa.parquet')  # It has to contain 'query' and 'generation_gt' columns.
llm = OpenAI(model='gpt-3.5-turbo', temperature=1.0)
qa_df = make_qa_with_existing_qa(corpus_df, existing_qa_df, content_size=50,
                                 exist_gen_gt=True, collection=collection,
                                 output_filepath='path/to/qa.parquet', cache_batch=64,
                                 embedding_model='openai_embed_3_large', top_k=5)
```
