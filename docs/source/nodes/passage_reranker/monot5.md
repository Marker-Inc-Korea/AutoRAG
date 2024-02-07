# MonoT5
The `MonoT5` module is a reranker that uses the MonoT5 model. This model rerank passages based on their relevance to a given query.

## **Module Parameters**
- (Optional) `model_name`:
  - Utilizes the monoT5 model for reranking, requiring the specification of a model_name. The model used must be defined in a provided model dictionary, allowing for flexible integration of different monoT5 variants.
  - default is `castorini/monot5-3b-msmarco-10k`

## **Example config.yaml**
```yaml
modules:
  - module_type: monot5
```