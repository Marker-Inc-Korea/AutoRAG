# Weaviate

The `Weaviate` class is an open-source vector database designed to store, query, and manage vector embeddings efficiently.

## Configuration

To use the Weaviate vector database, you need to configure it in your YAML configuration file.

### 1. Docker

You can see the full installation guide [here](https://weaviate.io/developers/weaviate/installation/docker-compose)

Set the `client_type` to `docker`.
You can run docker with the code below.
`docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.27.3`

It will automatically be set to host: localhost, port: 8080, and grpc_port: 50051.

If you're already running weaviate on docker, you'll need to set the host, port, and grpc_port you're using.

Additionally, you'll need to set `text_key`, which is the key value of the property where weaviate will store the content.
If you already have content stored in weaviate, you'll need to set that key value.

#### Example YAML file

```yaml
- name: openai_weaviate
  db_type: weaviate
  embedding_model: openai_embed_3_large
  collection_name: openai_embed_3_large
  client_type: docker
  host: localhost
  port: 8080
  grpc_port: 50051
  embedding_batch: 50
  similarity_metric: cosine
  text_key: content
```

Here is a simple example of a YAML configuration file that uses the Weaviate vector database and the OpenAI:

```yaml
vectordb:
  - name: openai_weaviate
    db_type: weaviate
    embedding_model: openai_embed_3_large
    collection_name: openai_embed_3_large
    client_type: docker
    host: localhost
    port: 8080
    grpc_port: 50051
    embedding_batch: 50
    similarity_metric: cosine
    text_key: content
node_lines:
- node_line_name: retrieve_node_line  # Arbitrary node line name
  nodes:
    - node_type: semantic_retrieval
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
      top_k: 3
      modules:
        - module_type: vectordb
          vectordb: openai_weaviate
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

### 2. Weaviate Cloud

You can see the full installation guide [here](https://weaviate.io/developers/weaviate/installation/weaviate-cloud-services)

#### Example YAML file

```yaml
- name: openai_embed_3_large
  db_type: weaviate
  embedding_model: openai_embed_3_large
  collection_name: openai_embed_3_large
  url: ${WEAVIATE_URL}
  api_key: ${WEAVIATE_API_KEY}
  grpc_port: 50051
  embedding_batch: 50
  similarity_metric: cosine
  text_key: content
```

Here is a simple example of a YAML configuration file that uses the Weaviate vector database and the OpenAI:

```yaml
vectordb:
  - name: openai_weaviate
    db_type: weaviate
    embedding_model: openai_embed_3_large
    collection_name: openai_embed_3_large
    url: ${WEAVIATE_URL}
    api_key: ${WEAVIATE_API_KEY}
    grpc_port: 50051
    embedding_batch: 50
    similarity_metric: cosine
    text_key: content
node_lines:
- node_line_name: retrieve_node_line  # Arbitrary node line name
  nodes:
    - node_type: semantic_retrieval
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
      top_k: 3
      modules:
        - module_type: vectordb
          vectordb: openai_weaviate
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

### Parameters

1. `embedding_model: str`
   - Purpose: Specifies the name or identifier of the embedding model to be used.
   - Example: "openai_embed_3_large"
   - Note: This should correspond to a valid embedding model that your system can use to generate vector embeddings. For more information see [custom your embedding model](https://marker-inc-korea.github.io/AutoRAG/local_model.html#configure-the-embedding-model) documentation.

2. `collection_name: str`
   - Purpose: Sets the name of the Weaviate collection where the vectors will be stored.
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
     - Not support "hamming", "manhattan"

5. `client_type = "docker"`
    - Purpose: Specifies the type of client you're using to connect to Weaviate.
    - Default: "docker"
    - Options: "docker", "cloud"
    - Note: Choose the appropriate client type based on your deployment.

6. `host: str = "localhost"`
   - Purpose: The host of the Weaviate server.
   - Default: "localhost"
   - Example: "weaviate-server.com"
   - Note: Use only `client_type: docker`.

7. `port: int = 8080`
    - Purpose: The port of the Weaviate Server.
    - Default: 8080
    - Note: Use only `client_type: docker`.

8. `grpc_port: int = 50051`
    - Note: Use only `client_type: docker`.

9. `url: str = ""`
   - Purpose: The URL of the Weaviate Cloud service.
   - Note: Use only `client_type: cloud`.

10. `api_key: str = ""`
    - Purpose: The API key for authentication with the Weaviate Cloud service.
    - Note: Use only `client_type: cloud`.

11. `text_key: str = "content"`
    - Purpose: Specifies the name of the property in Weaviate where the text data is stored.
    - Default: "content"
    - Note: This should correspond to the property name in your Weaviate schema where the text data is stored.

## Usage

Here's a brief overview of how to use the main functions of the Weaviate vector database:

1. **Adding Vectors**:
   ```python
   await weaviate_db.add(ids, texts)
   ```
   This method adds new vectors to the database. It takes a list of IDs and corresponding texts, generates embeddings, and inserts them into the Weaviate collection.

2. **Querying**:
   ```python
   ids, scores = await weaviate_db.query(queries, top_k)
   ```
   Performs a similarity search on the stored vectors.
   It returns the IDs and their scores.
   Below you can see how the score is determined.

3. **Fetching Vectors**:
   ```python
   vectors = await weaviate_db.fetch(ids)
   ```
   Retrieves the vectors associated with the given IDs.

4. **Checking Existence**:
   ```python
   exists = await weaviate_db.is_exist(ids)
   ```
   Checks if the given IDs exist in the database.

5. **Deleting Vectors**:
   ```python
   await weaviate_db.delete(ids)
   ```
   Deletes the vectors associated with the given IDs from the database.

6. **Deleting the Collection**:
   ```python
   weaviate_db.delete_collection()
   ```
   Deletes the collection from the Weaviate server.

### how the score is determined?

Calculate the score based on the distance.
[Here](https://weaviate.io/developers/weaviate/config-refs/distances) is how to calculate distance in Weaviate.

1. Cosine
    Score = 2 - distance

2. Inner Product
    Score = -distance

3. L2
    Score = -distance
