---
myst:
   html_meta:
      title: AutoRAG - OpenVINO Reranker
      description: Learn about openvino reranker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Reranker,OpenVINO Reranker
---
# OpenVINO Reranker
[OpenVINO™]() is an open-source toolkit for optimizing and deploying AI inference. The OpenVINO™ Runtime supports various hardware devices including x86 and ARM CPUs, and Intel GPUs. It can help to boost deep learning performance in Computer Vision, Automatic Speech Recognition, Natural Language Processing and other common tasks.

Hugging Face rerank model can be supported by OpenVINO through `OpenVINOReranker` class.

## **Module Parameters**

- **batch** : The size of a batch. If you have limited CUDA memory, decrease the size of the batch. (default: 64)
- **model** : The type of model id or path you want to use for reranking. Default is id "BAAI/bge-reranker-large"

## **Example config.yaml**

```yaml
- module_type: openvino_reranker
  batch: 32
  model: "BAAI/bge-reranker-large"
```
