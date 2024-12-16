---
myst:
   html_meta:
      title: AutoRAG - Cohere Reranker
      description: Learn about cohere reranker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Reranker,Cohere
---
# cohere_reranker

The `cohere reranker` module is a reranker from [cohere](https://cohere.ai/).
It supports powerful and fast reranker for passage retrieval.
Also, it supports multilingual languages.

## Before Usage

At first, you need to get the Cohere API key from [cohere](https://cohere.ai/).

Next, you can set your Cohere API key in the environment variable.

```bash
export COHERE_API_KEY=your_cohere_api_key
```

Or, you can set your Cohere API key in the config.yaml file directly.

```yaml
- module_type: cohere_reranker
  api_key: your_cohere_api_key
```

## **Module Parameters**

- **batch** : The size of a batch.
  It sends the batch size of passages to cohere API at once.
  If it is too large, it can cause some error.
  (default: 64)
- **model** : The type of model you want to use for reranking. Default is "rerank-multilingual-v2.0" and you can change
  it to "rerank-multilingual-v1.0" or "rerank-english-v2.0" (default: "rerank-multilingual-v2.0")
- **api_key** : The cohere api key.

## **Example config.yaml**

```yaml
- module_type: cohere_reranker
  api_key: your_cohere_api_key
  batch: 64
  model: rerank-multilingual-v2.0
```
