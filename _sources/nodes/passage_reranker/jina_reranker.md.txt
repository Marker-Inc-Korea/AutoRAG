---
myst:
   html_meta:
      title: AutoRAG - JINA Reranker
      description: Learn about JINA reranker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Reranker,JINA
---
# jina_reranker

The `Jina reranker` module is a reranker from [Jina](https://jina.ai/reranker).
It supports powerful and fast reranker for passage retrieval.

You can use two model from Jina reranker, which is 'jina-reranker-v1-base-en' and
'jina-colbert-v1-en'.

## Before Usage

At first, you need to get the Jina API key from [Jina](https://jina.ai/reranker).

Next, you can set your Jina API key in the environment variable.

```bash
export JINAAI_API_KEY=your_jina_api_key
```

Or, you can set your JinaAI API key in the config.yaml file directly.

```yaml
- module_type: jina_reranker
  api_key: your_jina_api_key
```

## **Module Parameters**

- **batch** : The size of a batch. It sends the batch size of passages to jina API at once. If it is too large, it can
  cause some error. (default: 8)
  You can increase when you have higher 'rpm' and 'tpm' limit from Jina AI.
- **model** : The type of model you want to use for reranking. Default is "jina-reranker-v1-base-en" and you can change
  it to "jina-reranker-v1-base-en" or "jina-colbert-v1-en"
- **api_key** : The cohere api key.

## **Example config.yaml**

```yaml
- module_type: jina_reranker
  api_key: your_jina_api_key
  batch: 16
  model: jina-colbert-v1-en
```
