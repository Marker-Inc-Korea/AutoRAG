# Vectordb

The `VectorDB` module is a retrieval module that uses VectorDB as a backend. You can use Dense Retrieval with this class easily.

It first embeds the passage content using an embedding model, then stores the embedded vector in VectorDB. When retrieving, it embeds the query and searches for the most similar vectors in VectorDB. Lastly, it returns the passages that have the most similar vectors.

### **Backend Support**

As of now, the `VectorDB` module exclusively supports **ChromaDB** as its backend database. ChromaDB is an open-source vector database designed to efficiently store and query embedding data, making it an ideal choice for applications involving Large Language Models (LLMs) and other embedding-based retrieval tasks.

## **Module Parameters**
- **Parameter**: `embedding_model`
- **Usage**: Defines the model used for embedding in the VectorDB module, impacting how data is represented and retrieved.
```{tip}
Information about the Embedding model can be found [Supporting Embedding models](../../local_model.md#supporting-embedding-models).
```

## **Example config.yaml**
```yaml
modules:
  - module_type: vectordb
    embedding_model: openai
```
