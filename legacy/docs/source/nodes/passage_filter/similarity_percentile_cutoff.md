---
myst:
   html_meta:
      title: AutoRAG - Similarity Percentile Cutoff
      description: Learn about similarity percentile cutoff passage filter module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Passage Filter,similarity percentile cutoff
---
# Similarity Percentile Cutoff

This module is inspired by
LlamaIndex ["Sentence Embedding Optimizer"](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/OptimizerDemo/).
Re-calculate each content's similarity with the query and filter out the contents that are below the content's
length times percentile.

## **Module Parameters**

- **percentile** : The percentile value to filter out the contents.
  This is essential to run the module, so you have to set this parameter.
- **embedding_model** : The embedding model name.
- **batch** : The batch size for embedding queries and contents.

```{tip}
Information about the Embedding model can be found [Supporting Embedding models](../../local_model.md#supporting-embedding-models).
Plus, you can learn about how to add custom embedding model at [here](../../local_model.md#add-your-embedding-models).
```

## **Example config.yaml**

```yaml
modules:
  - module_type: similarity_percentile_cutoff
    percentile: 0.6
    embedding_model: openai
    batch: 64
```
