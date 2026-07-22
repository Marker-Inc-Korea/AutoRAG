---
myst:
  html_meta:
    title: AutoRAG - Vector DB (MIPS)
    description: Learn about Vector DB module in AutoRAG
    keywords: AutoRAG,RAG,Advanced RAG,retrieval,Vector DB,MIPS
---
# Vectordb

The `VectorDB` module is a retrieval module that uses VectorDB as a backend. You can use Dense Retrieval with this class easily.

It first embeds the passage content using an embedding model, then stores the embedded vector in VectorDB. When retrieving, it embeds the query and searches for the most similar vectors in VectorDB. Lastly, it returns the passages that have the most similar vectors.

### **Backend Support**

As of now, the `VectorDB` module exclusively supports **ChromaDB** as its backend database.
We choose ChromaDB because it is a local VectorDB that needs no internet connection, server fee, or API key.
Plus, it is open-source software.

## **Module Parameters**
- **Parameter**: `vectordb`
- **Usage**: Defines the model used for embedding in the VectorDB module, impacting how data is represented and retrieved.
```{tip}
Information about the VectorDB can be found [VectorDB Section](../../vectordb/vectordb.md).
Plus, you can learn about how to add custom embedding model at [here](../../local_model.md#add-your-embedding-models).
```

## **Example config.yaml**
```yaml
modules:
  - module_type: vectordb
    vectordb: [default, openai_chroma, openai_milvus]
```
