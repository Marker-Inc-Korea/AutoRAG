---
myst:
  html_meta:
    title: AutoRAG - Vector DB (MIPS)
    description: Learn how to configure Vector DB in AutoRAG
    keywords: AutoRAG,RAG,Advanced RAG,retrieval,Vector DB,MIPS
---

# Configure Vector DB

## Overview

AutoRAG supports connection to multiple Vector Databases. This document outlines how to configure them.

## Supported Vector Databases

- Chroma (requires a link for setup)
- Milvus

## Usage

Configure your Vector DBs in the YAML file as shown below. For more detailed information on YAML file syntax, please refer to [here](https://docs.auto-rag.com/optimization/custom_config.html).

```yaml
vectordb:
  - name: openai_embed_3_small
    db_type: chroma
    client_type: persistent
    embedding_model: openai_embed_3_small
    collection_name: openai_embed_3_small
    path: ${PROJECT_DIR}/resources/chroma
  - name: openai_embed_3_large
    db_type: milvus
    embedding_model: openai_embed_3_large
    collection_name: openai_embed_3_large
    uri: ${MILVUS_URI}
    token: ${MILVUS_TOKEN}
    embedding_batch: 50
```

In this example, we've configured two vector databases:

1. The first uses the `openai_embed_3_small` model with Chroma. Chroma with a persistent client allows for simple local vector database usage.
2. The second uses Milvus. By providing the appropriate URI and token, you can access a remote Milvus instance. And it uses `openai_embed_3_large` as the embedding model.

For specific parameters and detailed configuration options, please refer to the respective documentation for each vector database.

## Full Ingest Option

The `full_ingest` option is crucial when using vector databases. By default, it's set to `True`. However, if your corpus is particularly large, it's recommended to set it to `False`.

When `full_ingest` is `True`, the system verifies that all IDs in the `corpus.parquet` file are properly stored in the vector database. This can be time-consuming and expensive for large datasets.

If set to `False`, the system only checks for the existence of IDs from the `retrieval_gt` in the QA data for each vector DB. This check cannot be disabled as the QA data is typically much smaller than the corpus, and crucial for successful trail on AutoRAG.

To set `full_ingest` to `False`, use one of the following methods:

### Command Line

```bash
autorag evaluate --config your/path/to/default_config.yaml --qa_data_path your/path/to/qa.parquet --corpus_data_path your/path/to/corpus.parquet --project_dir ./your/project/directory --full_ingest False
```

### Python Code

```python
from autorag.evaluator import Evaluator

evaluator = Evaluator(qa_data_path='your/path/to/qa.parquet', corpus_data_path='your/path/to/corpus.parquet',
                      project_dir='your/path/to/project_directory')
evaluator.start_trial('your/path/to/config.yaml', full_ingest=False)
```

## Default Options

AutoRAG provides a default Chroma vector database configuration if none is specified in the YAML file. In this case:

- Chroma will store data in the `project_dir/resources/chroma` folder
- The OpenAI `text-embedding-ada-002` model will be used for embeddings

You can use this default configuration by specifying it in the vectordb module of your YAML file:

```yaml
modules:
  - module_type: vectordb
    vectordb: default
```

This default option allows for quick setup and experimentation without the need for extensive configuration. However, for production use or specific requirements, it's recommended to explicitly configure your vector database settings.

### Additional Considerations

- If you having trouble at embedding phase, consider set `embedding_batch` parameter lower.
- Always ensure that your environment variables (like `${MILVUS_URI}` and `${MILVUS_TOKEN}`) are properly set when using remote databases.


```{toctree}
---
maxdepth: 1
---
chroma.md
milvus.md
weaviate.md
pinecone.md
couchbase.md
qdrant.md
```
