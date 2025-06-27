---
myst:
   html_meta:
      title: AutoRAG - Percentile Cutoff
      description: Learn about percentile cutoff passage filter module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Passage Filter,percentile cutoff
---
# Percentile Cutoff

This module is inspired by
our [similarity percentile cutoff](https://marker-inc-korea.github.io/AutoRAG/nodes/passage_filter/similarity_percentile_cutoff.html)
module.

Filter out the contents that are below the content's length times percentile.

## **Module Parameters**

- **percentile** : The percentile value to filter out the contents.
  This is essential to run the module, so you have to set this parameter.
- **reverse** : If True, the lower the score, the better.
  Default is False.

## **Example config.yaml**

```yaml
modules:
  - module_type: percentile_cutoff
    percentile: 0.6
```
