---
myst:
  html_meta:
    title: AutoRAG - hybrid rrf
    description: Learn about hybrid rrf module in AutoRAG
    keywords: AutoRAG,RAG,Advanced RAG,retrieval,hybrid rrf
---
# Hybrid - rrf

The `hybrid_rrf` module is designed to retrieve passages from multiple retrievals.
The `hybrid_rrf` module is tailored for retrieving passages from multiple sources of information.
It uses the Reciprocal Rank Fusion (RRF) algorithm to calculate final similarity scores.
This calculation is based on the ranking of passages in each retrieval,
effectively combining retrieval scores from different sources.

## ❗️Hybrid additional explanation

You can specify which rrf_k range that you want to explore. AutoRAG will find the optimal rrf_k parameter among your
specified range.
So, specify the range of rrf_k using `weight_range` is important to use hybrid_rrf.

## **Node Parameters**

- (Required) **top_k**: Essential parameter for retrieval node.

## **Module Parameters**

- (Optional) **weight_range**: The range of the weight(rrf_k) that you want to explore.
  The parameter name is `weight`, but it is actually `rrf_k` parameter at rrf algorithm.
  You have to input this value as tuple. It looks like this. `(10, 60)`. Default is `(4, 80)`.

## **Example config.yaml**
```yaml
modules:
  - module_type: hybrid_rrf
    weight_range: (4, 80)
```
