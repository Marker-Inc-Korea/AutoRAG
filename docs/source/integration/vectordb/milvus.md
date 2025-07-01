# Milvus

The `Milvus` class is a vector database implementation that allows you to store, query, and manage vector embeddings. It's designed to work with various embedding models and provides an efficient way to perform similarity searches.

## Configuration

To use the Milvus vector database, you need to configure it in your YAML configuration file. Here's an example configuration:

```yaml
- name: openai_milvus
  db_type: milvus
  embedding_model: openai_embed_3_large
  collection_name: openai_embed_3_large
  uri: ${MILVUS_URI}
  token: ${MILVUS_TOKEN}
  embedding_batch: 50
  similarity_metric: cosine
  index_type: IVF_FLAT
  params:
    nlist: 16384

```

Here is a simple example of a YAML configuration file that uses the Milvus vector database and the OpenAI:

```yaml
vectordb:
  - name: openai_milvus
    db_type: milvus
    embedding_model: openai_embed_3_large
    collection_name: openai_embed_3_large
    uri: ${MILVUS_URI}
    token: ${MILVUS_TOKEN}
    embedding_batch: 50
    similarity_metric: cosine
    index_type: IVF_FLAT
    params:
      nlist: 16384
node_lines:
- node_line_name: retrieve_node_line  # Arbitrary node line name
  nodes:
    - node_type: semantic_retrieval
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
      top_k: 3
      modules:
        - module_type: vectordb
          vectordb: openai_milvus
- node_line_name: post_retrieve_node_line  # Arbitrary node line name
  nodes:
    - node_type: prompt_maker
      strategy:
        metrics: [bleu, meteor, rouge]
      modules:
        - module_type: fstring
          prompt: "Read the passages and answer the given question. \n Question: {query} \n Passage: {retrieved_contents} \n Answer : "
    - node_type: generator
      strategy:
        metrics: [bleu, rouge]
      modules:
        - module_type: llama_index_llm
          llm: openai
          model: [ gpt-4o-mini ]
```

1. `embedding_model: str`
   - Purpose: Specifies the name or identifier of the embedding model to be used.
   - Example: "openai_embed_3_large"
   - Note: This should correspond to a valid embedding model that your system can use to generate vector embeddings. For more information see [custom your embedding model](https://marker-inc-korea.github.io/AutoRAG/local_model.html#configure-the-embedding-model) documentation.

2. `collection_name: str`
   - Purpose: Sets the name of the Milvus collection where the vectors will be stored.
   - Example: "my_vector_collection"
   - Note: If the collection doesn't exist, it will be created. If it exists, it will be loaded.

3. `embedding_batch: int = 100`
   - Purpose: Determines the number of embeddings to process in a single batch.
   - Default: 100
   - Note: Adjust this based on your system's memory and processing capabilities. Larger batches may be faster but require more memory.

4. `similarity_metric: str = "cosine"`
   - Purpose: Specifies the metric used to calculate similarity between vectors.
   - Default: "cosine"
   - Options: "cosine", "l2" (Euclidean distance), "ip" (Inner Product)
   - Note: Choose the metric that best suits your use case and data characteristics.

5. `uri: str = "http://localhost:19530"`
   - Purpose: The URI of the Milvus server.
   - Default: "http://localhost:19530"
   - Example: "http://milvus-server.com:19530"
   - Note: Use the appropriate URI for your Milvus server deployment.

6. `db_name: str = ""`
   - Purpose: Specifies the name of the database to use on the Milvus server.
   - Default: "" (empty string, uses the default database)
   - Note: Only set this if you're using multiple databases on your Milvus server.

7. `token: str = ""`
   - Purpose: Authentication token for the Milvus server.
   - Default: "" (empty string, no token)
   - Note: Set this if your Milvus server requires token-based authentication.

8. `user: str = ""`
   - Purpose: Username for authentication with the Milvus server.
   - Default: "" (empty string, no username)
   - Note: Set this if your Milvus server requires username/password authentication.

9. `password: str = ""`
   - Purpose: Password for authentication with the Milvus server.
   - Default: "" (empty string, no password)
   - Note: Set this along with the username if required for authentication.

10. `timeout: Optional[float] = None`
    - Purpose: Specifies the timeout duration (in seconds) for Milvus operations.
    - Default: None
    - Example: 30.0 (30 seconds timeout)
    - Note: Set this to control how long the client should wait for server responses before timing out.

#### Usage

Here's a brief overview of how to use the main functions of the Milvus vector database:

1. **Adding Vectors**:
   ```python
   await milvus_db.add(ids, texts)
   ```
   This method adds new vectors to the database. It takes a list of IDs and corresponding texts, generates embeddings, and inserts them into the Milvus collection.

2. **Querying**:
   ```python
   ids, distances = await milvus_db.query(queries, top_k)
   ```
   Performs a similarity search on the stored vectors. It returns the IDs of the most similar vectors and their distances.

3. **Fetching Vectors**:
   ```python
   vectors = await milvus_db.fetch(ids)
   ```
   Retrieves the vectors associated with the given IDs.

4. **Checking Existence**:
   ```python
   exists = await milvus_db.is_exist(ids)
   ```
   Checks if the given IDs exist in the database.

5. **Deleting Vectors**:
   ```python
   await milvus_db.delete(ids)
   ```
   Deletes the vectors associated with the given IDs from the database.

6. **Deleting the Collection**:
   ```python
   milvus_db.delete_collection()
   ```
   Deletes the collection from the Milvus server.
