---
myst:
   html_meta:
      title: AutoRAG - Time Reranker
      description: Learn about time reranker module in AutoRAG 
      keywords: AutoRAG,RAG,Advanced RAG,Reranker,TimeReranker
---
# Time Reranker

This reranker simply sorts the passages by their time.
It is useful when you want to use the latest information.
The time can be extracted from the corpus metadata.

You must have the `last_modified_datetime` field in the corpus metadata to use this reranker.
Plus, the value of the metadata must be `datetime.datetime` object.

## **Module Parameters**

- **Not Applicable (N/A):** There are no direct module parameters specified for the `time_reranker` module.

## **Example config.yaml**

```yaml
modules:
  - module_type: time_reranker
```
