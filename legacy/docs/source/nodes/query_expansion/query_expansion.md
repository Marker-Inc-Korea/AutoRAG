---
myst:
   html_meta:
      title: AutoRAG - Query Expansion
      description: Learn about query expansion module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,query expansion
---
# **1. Query Expansion**

### 🔎 **Definition**
Query expansion is the process of expanding a query without using it for immediate retrieval.
The goal is to improve the relevance of the search results by including variations of the query terms.


### 🤸 **Benefits**

- **Improved Response to Varied Queries**: Through query expansion, It’ll be better able to respond to different types of queries.
- **Increased RAG Accuracy**: By retrieving more relevant data through expanded queries, the accuracy of the generated output is significantly improved.

## 🔢 **Parameters**
### **Overview**:
This document serves as a guide for configuring parameters, strategies, and the YAML file.
### **Node Parameters**
- **Not Applicable (N/A):** There are no direct node parameters specified for the query expansion node.
### **Strategy**
**Performance Evaluation**: The result alone cannot measure the performance of the query_expansion node.
Therefore, it executes retrieval using queries, the result of Node, and evaluates it with the retrieval result.
Therefore, in strategy, we set the parameters necessary for retrieval and evaluation.
```{tip}
Please refer to the parameter of [retrieval Node](../retrieval/retrieval.md) for more details.
```

#### **Strategy Parameters**:

1. **Metrics**: Metrics such as `retrieval_f1`,`retrieval_recall`, and `retrieval_precision` are used to evaluate the performance of the query expansion process through its impact on retrieval outcomes.
2. **Speed Threshold**: `speed_threshold` is applied across all nodes, ensuring that any method exceeding the average processing time for a query is not used.
3. **Top_k**: This parameter specifies the number of top results to consider during the retrieval evaluation phase.
4. **Retrieval Modules**: The query expansion node can use modules and module parameters from the lexical retrieval and semantic retrieval node, including:
    - [bm25](../retrieval/bm25.md)
    - [vectordb](../retrieval/vectordb.md): with `vectordb` parameter
```{warning}
You cannot use the hybrid retrieval modules in the query expansion node.
```

### Example config.yaml file
```yaml
node_lines:
- node_line_name: pre_retrieve_node_line  # Arbitrary node line name
  nodes:
    - node_type: query_expansion
      strategy:
        metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        speed_threshold: 10
        top_k: 10
        retrieval_modules:
          - module_type: bm25
          - module_type: vectordb
            vectordb: default
      modules:
        - module_type: pass_query_expansion
        - module_type: query_decompose
          llm: openai
          temperature: [0.2, 1.0]
        - module_type: hyde
          llm: openai
          max_token: 64
```

```{admonition} What is pass_query_expansion?
Its purpose is to test the performance that 'not using' any query expansion module.
Because it can be the better option that not using query expansion node.
So with this module, you can automatically test the performance of the retrieval without using any query expansion module.
```

#### Supported Modules

```{toctree}
---
maxdepth: 1
---
query_decompose.md
hyde.md
multi_query_expansion.md
```
