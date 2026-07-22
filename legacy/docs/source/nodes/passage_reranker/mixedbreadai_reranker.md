---
myst:
   html_meta:
      title: AutoRAG - Mixedbread Reranker
      description: Learn about Mixedbread reranker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Reranker,Mixedbread Reranker
---
# Mixedbread AI Reranker

The `Mixedbread AI Reranker` module is a reranker that uses the mixedbread-ai rerank model. This model rerank passages based on their relevance to a
given query.

## Before Usage

At first, you need to get the Mixedbread AI API key from [MixedbreadAI](https://www.mixedbread.ai/api-reference#quick-start-guide).

Next, you can set your Mixedbread AI API key in the environment variable.

```bash
export MXBAI_API_KEY=your_mixedbread_api_key
```

Or, you can set your Mixedbread AI API key in the config.yaml file directly.

```yaml
- module_type: mixedbreadai_reranker
  api_key: your_mixedbread_api_key
```

## **Module Parameters**

- (Optional) `model_name`:
    - Requiring the specification of a model_name.
    - default is `mixedbread-ai/mxbai-rerank-large-v1`
- api_key: The Mixedbread AI api key.

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
