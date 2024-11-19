# AutoRAG Workflow API Documentation

This API provides a complete workflow for AutoRAG operations, from project creation to evaluation. The API follows RESTful principles and uses JSON for request/response payloads.

## Authentication

The API uses Bearer token authentication. Include the token in the Authorization header:

## Core Components

### Project
- Represents a RAG workflow project
- Contains metadata, status, and configuration
- Unique identifier: `proj_*`

### Task
- Represents individual workflow operations
- Types: parse, chunk, qa, validate, evaluate
- Contains status, configuration, and results
- Tracks execution state and errors

## API Endpoints

### Project Management
- `POST /projects`
  - Create a new project
  - Required: `name`
  - Returns: Project object

### Workflow Operations

#### 1. Parsing
- `POST /projects/{project_id}/parse/start`
  - Start document parsing
  - Required:
    - `glob_path`: File pattern to match
    - `config`: Parsing configuration
    - `name`: Operation name

#### 2. Chunking
- `POST /projects/{project_id}/chunk/start`
  - Process parsed documents into chunks
  - Required:
    - `raw_filepath`: Path to parsed data
    - `config`: Chunking configuration
    - `name`: Operation name

#### 3. QA Generation
- `POST /projects/{project_id}/qa/start`
  - Generate QA pairs
  - Required:
    - `corpus_filepath`: Path to chunked data
    - Optional:
      - `qa_num`: Number of QA pairs
      - `preset`: [basic, simple, advanced]
      - `llm_config`: LLM configuration

#### 4. Validation
- `POST /projects/{project_id}/validate/start`
  - Validate generated QA pairs
  - Required:
    - `config_yaml`: Validation configuration

#### 5. Evaluation
- `POST /projects/{project_id}/evaluate/start`
  - Evaluate RAG performance
  - Required:
    - `config_yaml`: Evaluation configuration
  - Optional:
    - `skip_validation`: Skip validation step (default: true)

### Task Monitoring
- `GET /projects/{project_id}/task/{task_id}`
  - Monitor task status
  - Returns: Task object with current status

## Task States
- `not_started`: Task is created but not running
- `in_progress`: Task is currently executing
- `completed`: Task finished successfully
- `failed`: Task failed with error

## Log Levels
- `info`: General information
- `warning`: Warning messages
- `error`: Error messages

## Typical Workflow Sequence
1. Create a project
2. Start parsing documents
3. Process chunks
4. Generate QA pairs
5. Validate results
6. Run evaluation
7. Monitor tasks through the task endpoint

## Response Formats

All successful responses return either a Project or Task object. Error responses include appropriate HTTP status codes and error messages.
