---
myst:
   html_meta:
      title: AutoRAG - Passage Reranker
      description: Learn about reranker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Reranker
---
# 4. Passage_Reranker

### 🔎 **Definition**
Reranking is a process applied after the initial retrieval of contents. It involves reassessing and reordering the contents based on their relevance to the query. This step is crucial in information retrieval systems to refine the results obtained from the first retrieval phase.


### 🤸 **Benefits**
The primary benefit of reranking is the enhanced prioritization of high-relevance contents.
Reranking ensures that the most pertinent information is more accessible,
improving the overall quality and relevance of the retrieved data.

## 🔢 **Parameters**
### **Overview**:
This document serves as a guide for configuring parameters, strategies, and the config YAML file for various nodes within a system.
### **Node Parameters**
**Top_k**
- **Description**: The `top_k` parameter is utilized at the node level to define the top 'k' results to be result passage size.

### **Strategy Parameters**
1. **Metrics**: The performance of the reranker is evaluated using metrics such as `retrieval_f1`, `retrieval_recall`, and `retrieval_precision`. These metrics assess the effectiveness of the reranking process in identifying the most relevant content.

2. **Speed Threshold**: An optional parameter that can be set to ensure the reranking process does not exceed a predefined processing time threshold.


### Example config.yaml file
```yaml
node_lines:
- node_line_name: retrieve_node_line  # Arbitrary node line name
  nodes:
  - node_type: passage_reranker
    strategy:
      metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
      speed_threshold: 10
    top_k: 5
    modules:
      - module_type: pass_reranker
      - module_type: tart
      - module_type: monot5
      - module_type: upr
```

```{admonition} What is pass_reranker?
Its purpose is to test the performance that 'not using' any passage reranker module.
Because it can be the better option that not using passage reranker node.
So with this module, you can automatically test the performance without using any passage reranker module.
```

#### Supported Modules

```{toctree}
---
maxdepth: 1
---
upr.md
tart.md
monot5.md
koreranker.md
cohere.md
rankgpt.md
jina_reranker.md
colbert.md
sentence_transformer_reranker.md
flag_embedding_reranker.md
flag_embedding_llm_reranker.md
time_reranker.md
openvino_reranker.md
voyageai_reranker.md
mixedbreadai_reranker.md
```
