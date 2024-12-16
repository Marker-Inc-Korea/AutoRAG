---
myst:
  html_meta:
    title: AutoRAG - Flag Embedding Reranker
    description: Learn about flag embedding reranker module in AutoRAG
    keywords: AutoRAG,RAG,Advanced RAG,Reranker,FlagEmbedding
---
# Flag Embedding LLM Reranker

The `flag embedding llm reranker` module is a reranker using BAAI LLM-based-Reranker model for
passage reranking.

## **Module Parameters**

- **batch** : The size of a batch. If you have limited CUDA memory, decrease the size of the batch. (default: 64)
- **model_name** : The type of model you want to use for reranking. Default is "BAAI/bge-reranker-v2-gemma."
    - you can check a model list at [here](https://github.com/FlagOpen/FlagEmbedding)
- **use_fp16** : Whether to use fp16 or not. (default: False)

## **Example config.yaml**

```yaml
- module_type: flag_embedding_llm_reranker
  batch: 32
  model_name: BAAI/bge-reranker-v2-gemma
```
