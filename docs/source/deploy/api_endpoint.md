---
myst:
   html_meta:
      title: AutoRAG - Deploy API endpoint
      description: Learn how to deploy optimized RAG pipeline to Flask API server in AutoRAG
      keywords: AutoRAG,RAG,RAG deploy,RAG API,Flask
---
# API endpoint

## Running API server

As mentioned in the tutorial, you can run api server as follows:

```python
from autorag.deploy import ApiRunner
import nest_asyncio

nest_asyncio.apply()

runner = ApiRunner.from_yaml('your/path/to/pipeline.yaml', project_dir='your/project/directory')
runner.run_api_server()
```

or

```python
from autorag.deploy import ApiRunner
import nest_asyncio

nest_asyncio.apply()

runner = ApiRunner.from_trial_folder('/your/path/to/trial_dir')
runner.run_api_server()
```

```bash
autorag run_api --trial_dir /trial/dir/0 --host 0.0.0.0 --port 8000
```

## Use NGrok Tunnel for public access

For accessing the API server from the public, you can use the NGrok tunnel service.
It automatically creates ngrok tunnel to your local server.

You can see the logs of the public URL like below:

```
INFO     [api.py:199] >> Public API URL:          api.py:199
         https://8a31-14-52-132-205.ngrok-free.app
```
This is the URL to your local server, so use it as the host at request.


## Endpoints

### 1. `/v1/run` (POST)

- **Summary**: Run a query and get generated text with retrieved passages.
- **Request Body**:
  - **Content Type**: `application/json`
  - **Schema**:
    - **Properties**:
      - `query` (string, required): The query string.
      - `result_column` (string, optional): The result column name. Default is `generated_texts`.
- **Responses**:
  - **200 OK**:
    - **Content Type**: `application/json`
    - **Schema**:
      - **Properties**:
        - `result` (string or array of strings): The result text or list of texts.
        - `retrieved_passage` (array of objects): List of retrieved passages.
          - **Properties**:
            - `content` (string): The content of the passage.
            - `doc_id` (string): Document ID.
            - `filepath` (string, nullable): File path.
            - `file_page` (integer, nullable): File page number.
            - `start_idx` (integer, nullable): Start index.
            - `end_idx` (integer, nullable): End index.

---

### 2. `/v1/retrieve` (POST)

This API endpoint allows developers to retrieve documents based on a specified query.
It will ignore generator and prompt maker, only return retrieved passages.

The request must include a JSON object with the following structure:

```json
{
  "query": "your query string here"
}
```

#### Parameters
- **query** (string, required): The search string used to retrieve documents.

### Example Request
```json
{
  "query": "latest trends in AI"
}
```

#### Success Response
**HTTP Status Code:** `200 OK`

#### Response Body
On a successful retrieval, the response will contain a JSON object structured as follows:

```json
{
  "passages": [
    {
      "doc_id": "unique-document-id-1",
      "content": "Content of the retrieved document.",
      "score": 0.95
    },
    {
      "doc_id": "unique-document-id-2",
      "content": "Content of another retrieved document.",
      "score": 0.89
    }
  ]
}
```

#### Properties
- **passages** (array): An array of documents retrieved based on the query.
  - **doc_id** (string): The unique identifier for each document.
  - **content** (string): The content of the retrieved document.
  - **score** (number, float): The relevance score of the retrieved document.

#### Example Response
```json
{
  "passages": [
    {
      "doc_id": "doc123",
      "content": "Artificial Intelligence is transforming industries.",
      "score": 0.98
    },
    {
      "doc_id": "doc456",
      "content": "The future of AI includes advancements in machine learning.",
      "score": 0.92
    }
  ]
}
```

---

### 3. `/v1/stream` (POST)

- **Summary**: Stream generated text with retrieved passages.
- **Description**: This endpoint streams the generated text line by line. The `retrieved_passage` is sent first, followed by the `result` streamed incrementally.
- **Request Body**:
  - **Content Type**: `application/json`
  - **Schema**:
    - **Properties**:
      - `query` (string, required): The query string.
      - `result_column` (string, optional): The result column name. Default is `generated_texts`.
- **Responses**:
  - **200 OK**:
    - **Content Type**: `text/event-stream`
    - **Schema**:
      - **Properties**:
        - `type` (generated_text or retrieved_passage): If it is generated_text, you can see only the generated text. If it is retrieved_passage, you can see the retrieved passage and passage_index.
        - `generated_text` (string): The generated text from the generator (LLM). The result of the RAG system.
        - `retrieved_passage` (object): Retrieved passage.
          - **Properties**:
            - `content` (string): The content of the passage.
            - `doc_id` (string): Document ID.
            - `filepath` (string, nullable): File path.
            - `file_page` (integer, nullable): File page number.
            - `start_idx` (integer, nullable): Start index.
            - `end_idx` (integer, nullable): End index.
        - `passage_index` (integer): Index of the retrieved passage.

---

### 4. `/version` (GET)

- **Summary**: Get the API version.
- **Description**: Returns the current version of the API as a string.
- **Responses**:
  - **200 OK**:
    - **Content Type**: `application/json`
    - **Schema**:
      - **Properties**:
        - `version` (string): The version of the API.

---

## API client usage example

Certainly! Below, I'll provide both Python sample code using the `requests` library and a `curl` command for each of the API endpoints described in the OpenAPI specification.

### Python Sample Code

First, ensure you have the `requests` library installed. You can install it using pip if you haven't already:

```bash
pip install requests
```

Here's the Python client code for each endpoint:

```python
import requests
from autorag.utils.util import decode_multiple_json_from_bytes

# Base URL of the API
BASE_URL = "http://example.com:8000"  # Replace with the actual base URL of the API

def run_query(query, result_column="generated_texts"):
    url = f"{BASE_URL}/v1/run"
    payload = {
        "query": query,
        "result_column": result_column
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def stream_query(query, result_column="generated_texts"):
    url = f"{BASE_URL}/v1/stream"
    payload = {
        "query": query,
        "result_column": result_column
    }
    with requests.Session() as session:
      response = session.post(url, json=payload, stream=True)
      retrieved_passages = [] # This will store retrieved passages

      # Check if the request was successful
      if response.status_code == 200:
          # Process the streaming response
          for i, chunk in enumerate(response.iter_content(chunk_size=None)):
              if chunk:
                  data_list = decode_multiple_json_from_bytes(chunk)
                  for data in data_list:
                      if data["type"] == "retrieved_passage":
                          retrieved_passages.append(data["retrieved_passage"])
                      else:
                          print(data["generated_text"], end="") # Stream the generated texts
      else:
          print(f"Request failed with status code: {response.status_code}")
          print(f"Response content: {response.text}")

def get_version():
    url = f"{BASE_URL}/version"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

# Example usage
if __name__ == "__main__":
    # Run a query
    result = run_query("example query")
    print("Run Query Result:", result)

    # Stream a query
    print("Stream Query Result:")
    stream_query("example query")

    # Get API version
    version = get_version()
    print("API Version:", version)
```

### `curl` Commands

Here are the equivalent `curl` commands for each endpoint:

#### `/v1/run` (POST)

```bash
curl -X POST "http://example.com/v1/run" \
     -H "Content-Type: application/json" \
     -d '{"query": "example query", "result_column": "generated_texts"}'
```

#### `/v1/stream` (POST)

```bash
curl -X POST "http://example.com/v1/stream" \
     -H "Content-Type: application/json" \
     -d '{"query": "example query", "result_column": "generated_texts"}' \
     --no-buffer
```

#### `/version` (GET)

```bash
curl -X GET "http://example.com/version"
```
