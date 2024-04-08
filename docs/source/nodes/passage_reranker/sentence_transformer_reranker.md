# Sentence Transformer Reranker

The `sentence transformer reranker` module is a reranker using sentence transformer model for
passage reranking.

## **Module Parameters**

- **batch** : The size of batch. If you have limited CUDA memory, decrease the size of the batch. (default: 64)
- **model_name** : The type of model you want to use for reranking. Default is "cross-encoder/ms-marco-MiniLM-L-2-v2".
- **sentence_transformer_max_length** : The maximum length of the input text. (default: 512)

## **Example config.yaml**

```yaml
- module_type: sentence_transformer_reranker
  batch: 32
  model_name: cross-encoder/ms-marco-MiniLM-L-2-v2
```