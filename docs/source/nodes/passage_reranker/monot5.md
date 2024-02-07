# MonoT5

The `MonoT5` module is a reranker that uses the MonoT5 model. This model rerank passages based on their relevance to a
given query.

## **Module Parameters**

- (Optional) `model_name`:
    - Requiring the specification of a model_name. The model used must be defined in a provided model dictionary,
      allowing for flexible integration of different monoT5 variants.
    - default is `castorini/monot5-3b-msmarco-10k`

## **Example config.yaml**

```yaml
modules:
  - module_type: monot5
```

### Supported Model Names

|                 Model Name                 |
|:------------------------------------------:|
|       castorini/monot5-base-msmarco        |
|     castorini/monot5-base-msmarco-10k      |
|       castorini/monot5-large-msmarco       |
|     castorini/monot5-large-msmarco-10k     |
|     castorini/monot5-base-med-msmarco      |
|      castorini/monot5-3b-med-msmarco       |
|      castorini/monot5-3b-msmarco-10k       |
|       unicamp-dl/mt5-base-en-msmarco       |
|   unicamp-dl/ptt5-base-pt-msmarco-10k-v2   |
|  unicamp-dl/ptt5-base-pt-msmarco-100k-v2   |
| unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2 |
|    unicamp-dl/mt5-base-en-pt-msmarco-v2    |
|       unicamp-dl/mt5-base-mmarco-v2        |
|    unicamp-dl/mt5-base-en-pt-msmarco-v1    |
|       unicamp-dl/mt5-base-mmarco-v1        |
|   unicamp-dl/ptt5-base-pt-msmarco-10k-v1   |
|  unicamp-dl/ptt5-base-pt-msmarco-100k-v1   |
| unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1  |
|       unicamp-dl/mt5-3B-mmarco-en-pt       |
|       unicamp-dl/mt5-13b-mmarco-100k       |
