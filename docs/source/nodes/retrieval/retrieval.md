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

## Three Node Types

```{admonition} Version Check
From AutoRAG v0.3.17, the retrieval node now divides into three types: `lexical_retrieval`, `semantic_retrieval` and `hybrid_retrieval`.
```

For the better usage and easier to implement new module, from AutoRAG v0.3.17, the retrieval node is divided into three types: `lexical_retrieval`, `semantic_retrieval`, and `hybrid_retrieval`.
- **Lexical Retrieval**: This node type is used for traditional keyword-based retrieval methods, such as BM25.
- **Semantic Retrieval**: This node type is used for retrieval methods that leverage semantic understanding, such as vector databases.
- **Hybrid Retrieval**: This node type combines both lexical and semantic retrieval methods, allowing for more comprehensive search capabilities.

So you need to define **three node types** in your config YAML file for using all retrievals. You must define both `lexical_retrieval` and `semantic_retrieval` nodes to use hybrid retrieval.
You can see the example config YAML files for more details.

### Example config.yaml file
```yaml
- node_line_name: retrieve_node_line
  nodes:
    - node_type: lexical_retrieval
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        speed_threshold: 10
      top_k: 10
      modules:
        - module_type: bm25
    - node_type: semantic_retrieval
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        speed_threshold: 10
      top_k: 10
      modules:
          - module_type: vectordb
            vectordb: default
    - node_type: hybrid_retrieval
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        speed_threshold: 10
      top_k: 10
      modules:
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
