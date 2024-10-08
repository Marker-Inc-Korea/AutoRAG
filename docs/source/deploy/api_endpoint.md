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

## API Endpoint

Certainly! To generate API endpoint documentation in Markdown format from the provided OpenAPI specification, we need to break down each endpoint and describe its purpose, request parameters, and response structure. Here's how you can document the API:

---

## Example API Documentation

### Version: 1.0.0

---

### Endpoints

#### 1. `/v1/run` (POST)

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

#### 2. `/v1/stream` (POST)

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
        - `result` (string or array of strings): The result text or list of texts (streamed line by line).
        - `retrieved_passage` (array of objects): List of retrieved passages.
          - **Properties**:
            - `content` (string): The content of the passage.
            - `doc_id` (string): Document ID.
            - `filepath` (string, nullable): File path.
            - `file_page` (integer, nullable): File page number.
            - `start_idx` (integer, nullable): Start index.
            - `end_idx` (integer, nullable): End index.

---

#### 3. `/version` (GET)

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
import json

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
    response = requests.post(url, json=payload, stream=True)
    if response.status_code == 200:
        for i, chunk in enumerate(response.iter_content(chunk_size=None)):
            if chunk:
                # Decode the chunk and print it
                data = json.loads(chunk.decode("utf-8"))
                if i == 0:
                    retrieved_passages = data["retrieved_passage"] # The retrieved passages
                print(data["result"], end="")
    else:
        response.raise_for_status()

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
