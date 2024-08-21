---
myst:
   html_meta:
      title: AutoRAG - Retrieval
      description: Learn about retrieval module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,retrieval
---
# 2. Retrieval

### 🔎 **Definition**
The retrieval process involves using queries to fetch relevant content, identifiers (IDs), and scores from a corpus. This is a fundamental operation in RAG, where the aim is to find the most relevant information based on the user's query.

## 🔢 **Parameters**

### **Overview**
This document serves as a guide for configuring parameters, strategies, and the YAML file for various nodes within a system.

### **Node Parameters**
**Top_k**
- **Description**: The `top_k` parameter is used at the node level to define the top 'k' results to be retrieved from corpus.

### **Strategy Parameters**
1. **Metrics**:
   - **Types**: `retrieval_f1`, `retrieval_recall`, `retrieval_precision`
   ```{admonition} Purpose
   These metrics are used to evaluate the effectiveness of the retrieval process, measuring the accuracy, recall, and precision of the retrieved content.
   ```

2. **Speed Threshold**:
   - **Description**: `speed_threshold` is applied across all nodes, ensuring that any method exceeding the average processing time for a query is not used.

### Example config.yaml file
```yaml
- node_line_name: retrieve_node_line
  nodes:
    - node_type: retrieval
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        speed_threshold: 10
      top_k: 10
      modules:
        - module_type: bm25
        - module_type: vectordb
          embedding_model: openai
        - module_type: hybrid_rrf
          weight_range: (4, 80)
        - module_type: hybrid_cc
          normalize_method: [ mm, tmm, z, dbsf ]
          weight_range: (0.0, 1.0)
          test_weight_size: 51
```

#### Supported Modules

```{toctree}
---
maxdepth: 1
---
bm25.md
vectordb.md
hybrid_rrf.md
hybrid_cc.md
```
