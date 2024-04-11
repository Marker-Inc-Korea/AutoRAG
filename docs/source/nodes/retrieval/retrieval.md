# 2. Retrieval

### 🔎 **Definition**
The retrieval process involves using queries to fetch relevant content, identifiers (IDs), and scores from a corpus. This is a fundamental operation in RAG, where the aim is to find the most relevant information based on the user's query.

## 🔢 **Parameters**

### **Overview**
This document serves as a guide for configuring parameters, strategies, and the YAML file for various nodes within a system.

### **Node Parameters**
**Top_k**
- **Description**: The `top_k` parameter is utilized at the node level to define the top 'k' results to be retrieved from corpus.

### **Strategy Parameters**
1. **Metrics**:  
   - **Types**: `retrieval_f1`, `retrieval_recall`, `retrieval_precision`
   ```{admonition} Purpose
   These metrics are used to evaluate the effectiveness of the retrieval process, measuring the accuracy, recall, and precision of the retrieved content.
   ```

2. **Speed Threshold**:
   - **Description**: `speed_threshold` is applied across all nodes, ensuring that any method exceeding the average processing time for a query is not utilized.

### Example config.yaml file
```yaml
- node_line_name: retrieve_node_line  # Arbitrary node line name
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
          target_modules: ('bm25', 'vectordb')
          rrf_k: [3, 5, 10]
        - module_type: hybrid_cc
          target_modules: ('bm25', 'vectordb')
          weights:
            - (0.5, 0.5)
            - (0.3, 0.7)
            - (0.7, 0.3)
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
hybrid_rsf.md
hybrid_dbsf.md
```
