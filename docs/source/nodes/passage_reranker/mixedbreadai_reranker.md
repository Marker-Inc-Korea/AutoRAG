---
myst:
   html_meta:
      title: AutoRAG - Mixedbread Reranker
      description: Learn about cohere reranker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Reranker,Mixedbread Reranker
---
# Mixedbread AI Reranker

The `Mixedbread AI Reranker` module is a reranker that uses the mixedbread-ai rerank model. This model rerank passages based on their relevance to a
given query.

## **Module Parameters**

- (Optional) `model_name`:
    - Requiring the specification of a model_name.
    - default is `mixedbread-ai/mxbai-rerank-large-v1`

## **Example config.yaml**

```yaml
modules:
  - module_type: mixedbreadai_reranker
```

### Supported Model Names

|                 Model Name                 |
|:------------------------------------------:|
|       mixedbread-ai/mxbai-rerank-xsmall-v1      |
|     mixedbread-ai/mxbai-rerank-large-v1     |
|       mixedbread-ai/mxbai-rerank-base-v1      |
