# 2. Retrieval

### 🔎 **Definition**
The retrieval process involves using queries to fetch relevant content, identifiers (IDs), and scores from a corpus. This is a fundamental operation in information retrieval systems, where the aim is to find the most relevant information based on the user's query.

## 🔢 **Parameters**

### **Overview**
This document serves as a guide for configuring parameters, strategies, and the `Config.yaml` file for various nodes within a system. It focuses particularly on the query expansion node but also provides insights applicable to other nodes.

### **Node Parameters**
**Top_k**
- **Description**: The `top_k` parameter is utilized at the node level to define the top 'k' results to be considered or processed further. This parameter is critical in scenarios where only a subset of the best results is required from a larger set.

### **Strategy**
#### **Key Parameters**:
1. **Metrics**:  
   - **Types**: `retrieval_f1`, `retrieval_recall`, `retrieval_precision`
   ```{admonition} Purpose
   These metrics are used to evaluate the effectiveness of the retrieval process, measuring the accuracy, recall, and precision of the retrieved content.
   ```

2. **Speed Threshold**:
   - **Description**: This optional parameter can be applied to all nodes to ensure that the processing time for a method does not exceed a predefined threshold. It is vital for maintaining the efficiency of the system.

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
```
