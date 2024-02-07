# 3. Passage_Reranker

### 🔎 **Definition**
Reranking is a process applied after the initial retrieval of contents. It involves reassessing and reordering the contents based on their relevance to the query. This step is crucial in information retrieval systems to refine the results obtained from the first retrieval phase.


### 🤸 **Benefits**
The primary benefit of reranking is the enhanced prioritization of high-relevance contents. By performing a subsequent evaluation, reranking ensures that the most pertinent information is more accessible, improving the overall quality and relevance of the retrieved data. This process significantly aids in sifting through potentially vast amounts of information to highlight what is most useful for the query at hand.

## 🔢 **Parameters**
### **Overview**:
This document serves as a guide for configuring parameters, strategies, and the `Config.yaml` file for various nodes within a system. It focuses particularly on the query expansion node but also provides insights applicable to other nodes.
### **Node Parameters**
**Top_k**
- **Description**: The `top_k` parameter is utilized at the node level to define the top 'k' results to be considered or processed further. This parameter is critical in scenarios where only a subset of the best results is required from a larger set.

### **Strategy**

#### **Key Parameters**:
1. **Metrics**: The performance of the reranker is evaluated using metrics such as `retrieval_f1`, `retrieval_recall`, and `retrieval_precision`. These metrics assess the effectiveness of the reranking process in identifying the most relevant content.

2. **Speed Threshold**: An optional parameter that can be set to ensure the reranking process does not exceed a predefined processing time threshold. This parameter helps balance the trade-off between reranking accuracy and system performance.


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
      - module_type: tart
      - module_type: monot5
      - module_type: UPR
```

#### Supported Modules

```{toctree}
---
maxdepth: 1
---
upr.md
tart.md
monot5.md
```
