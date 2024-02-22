# Ko-reranker

The `ko-reranker` module is a reranker based on **korean**.
More details can be found [here](https://huggingface.co/Dongjin-kr/ko-reranker).


## **Module Parameters**
(Optional) `instruction`
- Specifies instructions for the reranking process. 
- default is `Find passage to answer given question`

## **Example config.yaml**
```yaml
modules:
  - module_type: koreranker
```
