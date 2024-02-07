# TART

The `TART` module is a reranker based on [TART](https://arxiv.org/pdf/2211.09260.pdf). It is designed to rerank passages using specific instructions. The primary functionality of this class lies in its ability to rerank a list of passages based on a given query and instruction.

## **Module Parameters**
(Optional) `instruction`
- Specifies instructions for the reranking process. 
- default is `Find passage to answer given question`

## **Example config.yaml**
```yaml
modules:
  - module_type: tart
```
