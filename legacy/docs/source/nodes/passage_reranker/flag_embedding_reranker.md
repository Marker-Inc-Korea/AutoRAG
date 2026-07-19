---
myst:
   html_meta:
      title: AutoRAG - Flag Embedding LLM Reranker
      description: Learn about flag embedding LLM reranker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Reranker,FlagEmbedding,FlagEmbeddingLLM
---
# Flag Embedding Reranker

The `flag embedding reranker` module is a reranker using BAAI normal-Reranker model for
passage reranking.

## **Module Parameters**

- **batch** : The size of a batch. If you have limited CUDA memory, decrease the size of the batch. (default: 64)
- **model_name** : The type of model you want to use for reranking. Default is "BAAI/bge-reranker-large."
    - you can check a model list at [here](https://github.com/FlagOpen/FlagEmbedding)
- **use_fp16** : Whether to use fp16 or not. (default: False)

## **Example config.yaml**

```yaml
- module_type: flag_embedding_reranker
  batch: 32
  model_name: BAAI/bge-reranker-large
```
