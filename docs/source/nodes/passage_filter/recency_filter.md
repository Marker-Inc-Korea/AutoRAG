---
myst:
   html_meta:
      title: AutoRAG - Passage Filter
      description: Learn about passage filter module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Passage Filter
---
# Recency Filter

This module is inspired by
LlamaIndex ["Recency Filtering"](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/RecencyPostprocessorDemo/)

Filter out the contents that are below the threshold datetime.
If all contents are filtered, keep the only one recency content.
If the threshold date format is incorrect, return the original contents.

It is useful when you want to use the latest information.
The time can be extracted from the corpus metadata.

You must have the `last_modified_datetime` field in the corpus metadata to use this reranker.
Plus, the value of the metadata must be `datetime.datetime` object.

## **Module Parameters**

- **threshold** : The threshold value to filter out the contents.
  If the time is later than a threshold, the content will be filtered out.
  This is essential to run the module, so you have to set this parameter.

  📌 **threshold** format should be one of the following three!
  - `YYYY-MM-DD`
  - `YYYY-MM-DD HH:MM`
  - `YYYY-MM-DD HH:MM:SS`

## **Example config.yaml**

```yaml
modules:
  - module_type: recency_filter
    threshold: 2015-01-01
```
