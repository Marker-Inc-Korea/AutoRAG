---
myst:
   html_meta:
      title: AutoRAG - Colbert Reranker
      description: Learn about colbert reranker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Reranker,colbert
---
# Colbert Reranker

The `colber reranker` module is a reranker using [ColBERT](https://huggingface.co/colbert-ir/colbertv2.0) model for
passage reranking.

## **Module Parameters**

- **batch** : The size of a batch. If you have limited CUDA memory, decrease the size of the batch. (default: 64)
- **model_name** : The type of model you want to use for reranking. Default is "colbert-ir/colbertv2.0".

## **Example config.yaml**

```yaml
- module_type: colbert_reranker
  batch: 64
  model_name: colbert-ir/colbertv2.0
```
