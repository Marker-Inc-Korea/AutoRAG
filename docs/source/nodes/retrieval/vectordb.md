# Vectordb

The `VectorDB` module is a retrieval module that uses VectorDB as a backend. You can use Dense Retrieval with this class easily.

It first embeds the passage content using an embedding model, then stores the embedded vector in VectorDB. When retrieving, it embeds the query and searches for the most similar vectors in VectorDB. Lastly, it returns the passages that have the most similar vectors.

### **Backend Support**

As of now, the `VectorDB` module exclusively supports **ChromaDB** as its backend database.
We choose ChromaDB because it is a local VectorDB that needs no internet connection, server fee, or API key.
Plus, it is open-source software. 

## **Module Parameters**
- **Parameter**: `embedding_model`
- **Usage**: Defines the model used for embedding in the VectorDB module, impacting how data is represented and retrieved.
```{tip}
Information about the Embedding model can be found [Supporting Embedding models](../../local_model.md#supporting-embedding-models).
Plus, you can learn about how to add custom embedding model at [here](../../local_model.md#add-your-embedding-models). 
```

- **Parameter**: `embedding_batch`
- **Usage**: It is the batch size of the embedding model. It automatically set to the ingestion process using embedding
  model.
  If you get error on the embedding model, try to lower this parameter.

## **Example config.yaml**
```yaml
modules:
  - module_type: vectordb
    embedding_model: openai
    embedding_batch: 64
```
