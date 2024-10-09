---
myst:
   html_meta:
      title: AutoRAG - VoyageAI Reranker
      description: Learn about voyage ai reranker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Reranker,VoyageAI
---
# voyageai_reranker

The `voyageai reranker` module is a reranker from [VoyageAI](https://www.voyageai.com/).
It supports powerful and fast reranker for passage retrieval.

## Before Usage

At first, you need to get the VoyageAI API key from [here](https://docs.voyageai.com/docs/api-key-and-installation).

Next, you can set your VoyageAI API key in the environment variable "VOYAGE_API_KEY".

```bash
export VOYAGE_API_KEY=your_voyageai_api_key
```

Or, you can set your VoyageAI API key in the config.yaml file directly.

```yaml
- module_type: voyageai_reranker
  api_key: your_voyageai_api_key
```

## **Module Parameters**

- **model** : The type of model you want to use for reranking. Default is "rerank-2" and you can change
  it to "rerank-2-lite"
- **api_key** : The voyageai api key.
- **truncation** : Whether to truncate the input to satisfy the 'context length limit' on the query and the documents. Default is True.

## **Example config.yaml**

```yaml
- module_type: voyageai_reranker
  api_key: your_voyageai_api_key
  model: rerank-2
```

### Supported Model Names

You can see the supported model names [here](https://docs.voyageai.com/docs/reranker).

|  Model Name   |
|:-------------:|
|   rerank-2    |
| rerank-2-lite |
