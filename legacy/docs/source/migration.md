# Migration Guide

1. [v0.3 migration guide](#v03-migration-guide)

## v0.3 migration guide

### Data Creation

From the v0.3 version, the previous data creation library goes into the `legacy` package.
Instead of legacy data creation, the `beta` package is introduced.
There are no longer `beta` package at the data, and you can use it without `beta` import.

For example,

- v0.2 version

```python
from autorag.data.corpus import langchain_documents_to_parquet
from autorag.data.qacreation import generate_qa_llama_index, make_single_content_qa
```

```python
from autorag.data.beta.query.llama_gen_query import factoid_query_gen
from autorag.data.beta.sample import random_single_hop
from autorag.data.beta.schema import Raw
```

- v0.3 version

```python
from autorag.data.legacy.corpus import langchain_documents_to_parquet
from autorag.data.legacy.qacreation import generate_qa_llama_index, make_single_content_qa
```

```python
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.sample import random_single_hop
from autorag.data.qa.schema import Raw
```

## v0.3.7 migration guide

At v0.3.6, there are changes of the vectordb.
You have to specify what vectordb you want to use at the config YAML file.

- v0.3.6 version (previous v0.3.7)

```yaml
node_lines:
- node_line_name: retrieve_node_line
  nodes:
    - node_type: retrieval  # represents run_node function
      strategy:  # essential for every node
        metrics: [retrieval_f1, retrieval_recall]
      top_k: 10 # node param, which adapt to every module in this node.
      modules:
        - module_type: bm25
          bm25_tokenizer: [ facebook/opt-125m, porter_stemmer ]
        - module_type: vectordb
          embedding_model: [openai_embed_3_large, openai_embed_3_small]
        - module_type: hybrid_rrf
          weight_range: (4, 30)
```

- v0.3.7 version

```yaml
vectordb:
  - name: openai_embed_3_small
    db_type: chroma
    client_type: persistent
    embedding_model: openai_embed_3_small
    collection_name: openai_embed_3_small
    path: ${PROJECT_DIR}/resources/chroma
  - name: openai_embed_3_large
    db_type: chroma
    client_type: persistent
    embedding_model: openai_embed_3_large
    collection_name: openai_embed_3_large
    path: ${PROJECT_DIR}/resources/chroma
    embedding_batch: 50
node_lines:
- node_line_name: retrieve_node_line
  nodes:
    - node_type: retrieval  # represents run_node function
      strategy:  # essential for every node
        metrics: [retrieval_f1, retrieval_recall]
      top_k: 10 # node param, which adapt to every module in this node.
      modules:
        - module_type: bm25
          bm25_tokenizer: [ facebook/opt-125m, porter_stemmer ]
        - module_type: vectordb
          vectordb: [openai_embed_3_large, openai_embed_3_small]
        - module_type: hybrid_rrf
          weight_range: (4, 30)
```

For more information about vectordb, you can refer to the [vectordb documentation](integration/vectordb/vectordb.md).


## v0.3.17 migration guide

From AutoRAG v0.3.17, the retrieval node now divides into three types: `lexical_retrieval`, `semantic_retrieval`, and `hybrid_retrieval`.
You need to split the previous retrieval node into three types.
You need to specify both 'strategy' and 'top_k' for every node type.

- v0.3.17 version
```yaml
vectordb:
  - name: openai_embed_3_small
    db_type: chroma
    client_type: persistent
    embedding_model: openai_embed_3_small
    collection_name: openai_embed_3_small
    path: ${PROJECT_DIR}/resources/chroma
  - name: openai_embed_3_large
    db_type: chroma
    client_type: persistent
    embedding_model: openai_embed_3_large
    collection_name: openai_embed_3_large
    path: ${PROJECT_DIR}/resources/chroma
    embedding_batch: 50
node_lines:
- node_line_name: retrieve_node_line
  nodes:
    - node_type: lexical_retrieval # Lexical retrieval node type
      strategy:
        metrics: [retrieval_f1, retrieval_recall]
      top_k: 10
      modules:
        - module_type: bm25
          bm25_tokenizer: [ facebook/opt-125m, porter_stemmer ]
    - node_type: semantic_retrieval # semantic retrieval node type
      strategy:
        metrics: [retrieval_f1, retrieval_recall]
      top_k: 10
      modules:
        - module_type: vectordb
          vectordb: [openai_embed_3_large, openai_embed_3_small]
    - node_type: hybrid_retrieval # Hybrid retrieval node type
      strategy:
        metrics: [retrieval_f1, retrieval_recall]
      top_k: 10
      modules:
        - module_type: hybrid_rrf
          weight_range: (4, 30)
        - module_type: hybrid_cc
          normalize_method: [ mm, tmm, z, dbsf ]
          weight_range: (0.0, 1.0)
          test_weight_size: 51
```

This YAML file do the same thing as the previous v0.3.7 version.

Also, you’re no longer able to use the hybrid retrieval node in the `query_expansion` node as `retrieval_modules`.
We’re considering to add this feature in the future, but for now, you can use semantic and lexical retrieval nodes to evaluate query expansion.
For most cases, you don't need to use hybrid retrieval node in the `query_expansion` node.
