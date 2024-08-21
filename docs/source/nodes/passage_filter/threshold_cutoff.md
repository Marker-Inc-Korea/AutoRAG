---
myst:
   html_meta:
      title: AutoRAG - Threshold Cutoff
      description: Learn about threshold cutoff passage filter module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Passage Filter,threshold cutoff
---
# Threshold Cutoff

This module is inspired by our [similarity threshold cutoff](https://docs.auto-rag.com/nodes/passage_filter/similarity_threshold_cutoff.html) module.
Filters the contents, scores, and ids based on a **previous result's scores**.

📣 Keeps at least one item per query if all scores are below the threshold.

## **Module Parameters**

- **threshold** : The threshold value to filter out the contents.
  If the score is below the threshold, the content will be filtered out.
  This is essential to run the module, so you have to set this parameter.
- **reverse** : If True, the lower the score, the better.
  Default is False.

## **Example config.yaml**

```yaml
modules:
  - module_type: threshold_cutoff
    threshold: 0.85
```
