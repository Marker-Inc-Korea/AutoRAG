---
myst:
  html_meta:
    title: AutoRAG - NVIDIA Reranker
    description: Learn about NVIDIA reranker module in AutoRAG
    keywords: AutoRAG,RAG,Advanced RAG,Reranker,NVIDIA
---

# nvidia_reranker

The `NVIDIA reranker` module is a reranker from [NVIDIA](https://docs.api.nvidia.com/nim/reference/nvidia-rerank-qa-mistral-4b).

You can use one model from NVIDIA reranker, which is 'nvidia/rerank-qa-mistral-4b'.

## Before Usage

At first, you need to get the NVIDIA API key from [NVIDIA API](https://build.nvidia.com/nvidia/rerank-qa-mistral-4b).

Next, you can set your NVIDIA API key in the environment variable.

```bash
export NVIDIA_API_KEY=your_nvidia_api_key
```

Or, you can set your NVIDIA API key in the config.yaml file directly.

```yaml
- module_type: nvidia_reranker
  api_key: your_nvidia_api_key
```

## **Module Parameters**

- **batch** : The size of a batch. It sends the batch size of queries to NVIDIA API at once. If it is too large, it can
  cause some error. (default: 64)
  You can adjust this value based on your API rate limits and performance requirements.
- **model** : The type of model you want to use for reranking. Default is "nvidia/rerank-qa-mistral-4b".
  Currently, only "nvidia/rerank-qa-mistral-4b" is tested and verified.
- **truncate** : Optional truncation strategy for the API request. If not specified, no truncation is applied.
  You can set this parameter to control how the API handles long inputs.
- **api_key** : The NVIDIA API key. If not provided, it will use the NVIDIA_API_KEY environment variable.

## **Example config.yaml**

```yaml
- module_type: nvidia_reranker
  api_key: your_nvidia_api_key
  batch: 32
  model: nvidia/rerank-qa-mistral-4b
```
