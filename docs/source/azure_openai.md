---
myst:
  html_meta:
    title: AutoRAG - Azure OpenAI Integration
    description: Learn how to use Azure OpenAI with AutoRAG for RAG pipeline optimization
    keywords: AutoRAG,RAG,Azure,Azure OpenAI,GPT,LLM,Microsoft,enterprise
---

# Azure OpenAI Integration

This guide explains how to use **Azure OpenAI** with AutoRAG. Azure OpenAI is widely adopted in enterprise
environments where organizations require Azure-only cloud services.

## Prerequisites

### 1. Install Dependencies

Install the required LlamaIndex Azure OpenAI packages:

```bash
pip install AutoRAG llama-index-llms-azure-openai llama-index-embeddings-azure-openai
```

### 2. Set Up Azure OpenAI Resource

Before using Azure OpenAI with AutoRAG, you need:

1. An **Azure OpenAI resource** in your Azure subscription
2. A **deployment** for your chosen model (e.g., `gpt-4o-mini`, `gpt-4o`, `text-embedding-3-small`)
3. The **API key** and **endpoint** from the Azure portal

You can find these in the Azure Portal under your Azure OpenAI resource → **Keys and Endpoint**.

### 3. Set Environment Variables

```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-01"
```

On Windows:
```powershell
$env:AZURE_OPENAI_API_KEY = "your-azure-openai-api-key"
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource-name.openai.azure.com/"
$env:AZURE_OPENAI_API_VERSION = "2024-02-01"
```

## Using Azure OpenAI as Generator (LLM)

AutoRAG supports Azure OpenAI through the `llama_index_llm` module with `llm: azure_openai`.

### Config YAML Example

```yaml
nodes:
  - node_line_name: post_retrieve_node_line
    nodes:
      - node_type: generator
        strategy:
          metrics: [bleu, rouge]
        modules:
          - module_type: llama_index_llm
            llm: azure_openai
            model: gpt-4o-mini
            engine: your-deployment-name  # Your Azure deployment name
            api_key: ${AZURE_OPENAI_API_KEY}
            azure_endpoint: ${AZURE_OPENAI_ENDPOINT}
            api_version: "2024-02-01"
```

```{important}
The `engine` parameter is the **deployment name** you set in the Azure Portal, not the model name.
The `model` parameter is the actual model (e.g., `gpt-4o-mini`, `gpt-4o`).
Both are required for Azure OpenAI.
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `llm` | Yes | Must be `azure_openai` |
| `model` | Yes | The model name (e.g., `gpt-4o-mini`, `gpt-4o`) |
| `engine` | Yes | Your Azure deployment name |
| `api_key` | Yes | Azure OpenAI API key (or set `AZURE_OPENAI_API_KEY` env var) |
| `azure_endpoint` | Yes | Azure OpenAI endpoint URL |
| `api_version` | No | API version string (default: `2024-02-01`) |
| `temperature` | No | Controls randomness (0.0 to 2.0, default: 0.1) |
| `max_tokens` | No | Maximum tokens in response |

## Using Azure OpenAI for Embeddings

For embeddings, you can use Azure OpenAI through the `openai_like` embedding type or configure it programmatically.

### Option 1: Using Environment Variables (Recommended)

Set the OpenAI environment variables to point to your Azure endpoint:

```bash
export OPENAI_API_KEY="your-azure-openai-api-key"
export OPENAI_API_BASE="https://your-resource-name.openai.azure.com/"
export OPENAI_API_VERSION="2024-02-01"
export OPENAI_API_TYPE="azure"
```

Then use the standard OpenAI embedding config:

```yaml
vectordb:
  - name: azure_chroma
    db_type: chroma
    client_type: persistent
    embedding_model: openai_embed_3_small
    collection_name: azure_embed
    path: ${PROJECT_DIR}/data/chroma
```

### Option 2: Programmatic Configuration

```python
import autorag
from autorag import LazyInit
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

autorag.embedding_models['azure_openai_embed'] = LazyInit(
    AzureOpenAIEmbedding,
    model="text-embedding-3-small",
    deployment_name="your-embedding-deployment",
    api_key="your-api-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-01",
)
```

Then reference it in your config:

```yaml
vectordb:
  - name: azure_chroma
    db_type: chroma
    client_type: persistent
    embedding_model: azure_openai_embed
    collection_name: azure_embed
    path: ${PROJECT_DIR}/data/chroma
```

## Full Example Config

A complete sample config file using Azure OpenAI is available at
[`sample_config/rag/english/non_gpu/simple_azure_openai.yaml`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/sample_config/rag/english/non_gpu/simple_azure_openai.yaml).

```yaml
vectordb:
  - name: azure_openai_chroma
    db_type: chroma
    client_type: persistent
    embedding_model: openai_embed_3_small
    collection_name: azure_openai_embed
    path: ${PROJECT_DIR}/data/chroma

node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: semantic_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        top_k: 3
        modules:
          - module_type: vectordb
            vectordb: azure_openai_chroma

  - node_line_name: post_retrieve_node_line
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
            llm: azure_openai
            model: gpt-4o-mini
            engine: your-gpt4o-mini-deployment
            api_key: ${AZURE_OPENAI_API_KEY}
            azure_endpoint: ${AZURE_OPENAI_ENDPOINT}
            api_version: "2024-02-01"
            temperature: 0.1
```

## Using Azure OpenAI with Query Expansion

Azure OpenAI can also be used in query expansion modules like `hyde`, `query_decompose`, and `multi_query_expansion`:

```yaml
- node_type: query_expansion
  modules:
    - module_type: hyde
      generator_module_type: llama_index_llm
      llm: azure_openai
      model: gpt-4o-mini
      engine: your-deployment-name
      api_key: ${AZURE_OPENAI_API_KEY}
      azure_endpoint: ${AZURE_OPENAI_ENDPOINT}
      api_version: "2024-02-01"
      max_token: 64
```

## Troubleshooting

### Common Issues

1. **`ImportError: No module named 'llama_index.llms.azure_openai'`**

   Install the Azure OpenAI package:
   ```bash
   pip install llama-index-llms-azure-openai
   ```

2. **`AuthenticationError` or `401 Unauthorized`**

   - Verify your API key is correct
   - Check that your Azure OpenAI resource is properly provisioned
   - Ensure your deployment exists and is active

3. **`DeploymentNotFound` or `404`**

   - The `engine` parameter must match your **deployment name** exactly (not the model name)
   - Check your deployment in Azure Portal → Azure OpenAI → Model deployments

4. **`RateLimitError` or `429`**

   - Azure OpenAI has per-deployment rate limits (TPM/RPM)
   - Reduce the `batch` parameter in your config
   - Consider using a higher-tier deployment

### Tips for Enterprise Users

- Use **Azure Managed Identity** for authentication instead of API keys when running in Azure
- Configure **Virtual Network** integration for private endpoints
- Use the latest `api_version` for the best feature support
- Monitor usage through **Azure Monitor** and set up alerts for rate limits
