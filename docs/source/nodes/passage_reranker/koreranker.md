---
myst:
   html_meta:
      title: AutoRAG - Ko-Reranker
      description: Learn about ko-reranker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Reranker,Ko-Reranker
---
# Ko-reranker

The `ko-reranker` module is a reranker based on **korean**.
More details can be found [here](https://huggingface.co/Dongjin-kr/ko-reranker).


## **Module Parameters**

(Optional) `batch`

- Specify the batch size of the query to the Ko-reranker model.
- default is 64.

## **Example config.yaml**
```yaml
modules:
  - module_type: koreranker
```
