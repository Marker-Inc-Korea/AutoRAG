---
myst:
   html_meta:
      title: AutoRAG - Similarity Threshold Cutoff
      description: Learn about similarity threshold cutoff passage filter module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Passage Filter
---
# Similarity Threshold Cutoff

This module is inspired by
LlamaIndex ["Sentence Embedding Optimizer"](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/OptimizerDemo/).
Re-calculate each content's similarity with the query and filter out the contents that are below the threshold.

📣 Keeps at least one item per query if all scores are below the threshold.

## **Module Parameters**

- **threshold** : The threshold value to filter out the contents.
  If the similarity score is below the threshold, the content will be filtered out.
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
  - module_type: similarity_threshold_cutoff
    threshold: 0.85
    embedding_model: openai
    batch: 64
```
