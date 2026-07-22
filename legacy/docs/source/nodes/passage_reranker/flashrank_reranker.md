---
myst:
   html_meta:
      title: AutoRAG - FlashRank Reranker
      description: Learn about flashrank reranker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Reranker,FlashRank Reranker
---
# FlashRank Reranker
[FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) is the Ultra-lite & Super-fast Python library to add re-ranking to your existing search & retrieval pipelines.

It is based on SoTA cross-encoders, with gratitude to all the model owners.

## **Module Parameters**

- **batch** : The size of a batch. If you have limited CUDA memory, decrease the size of the batch. (default: 64)
- **model** : The type of model id or path you want to use for reranking. Default is id ""ms-marco-TinyBERT-L-2-v2"".
  - You can get the list of available models from [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank.)
  ```{admonition} Note
  “rank_zephyr_7b_v1_full” is an llm based reranker that uses llama-cpp.
  Due to issues with parallel inference, “rank_zephyr_7b_v1_full” is not currently supported by AutoRAG.
  ```

## **Example config.yaml**

```yaml
- module_type: flashrank_reranker
  batch: 32
  model: "ms-marco-MiniLM-L-12-v2"
```
