# 6. Generator

### 🔎 **Definition**
**`Generator Node`:** A component within a system that allows for experimentation with various Large Language Models (LLMs). This node is designed to facilitate the testing and evaluation of different LLMs to determine the most effective model for specific tasks or data sets.

## 🔢 **Parameters**

### **Overview**
This document serves as a guide for configuring parameters, strategies, and the `Config.yaml` file for various nodes within a system. It focuses particularly on the query expansion node but also provides insights applicable to other nodes.

### **Node Parameters**
- **Not Applicable (N/A):** There are no direct node parameters specified for the query expansion node, indicating a focus on strategy-based configuration.

### **Strategy**
#### **Key Parameters**:
1. **Metrics**:  
   - **Types**: `bleu`, `meteor`, `rouge`
   ```{admonition} Purpose
   These metrics are used to evaluate the performance of language models by comparing model-generated text to human reference texts, assessing translation quality, summarization, and understanding
   ```

2. **Speed Threshold**:
   - **Description**: This optional parameter can be applied to all nodes to ensure that the processing time for a method does not exceed a predefined threshold. It is vital for maintaining the efficiency of the system.

### Example config.yaml file
```yaml
- node_line_name: post_retrieve_node_line  # Arbitrary node line name
  nodes:
    - node_type: generator
      strategy:
        metrics: [bleu, meteor, rouge]
        speed_threshold: 10
      modules:
        - module_type: llama_index_llm
          llm: [openai]
          model: [gpt-3.5-turbo-16k, gpt-3.5-turbo-1106]
          temperature: [0.5, 1.0, 1.5]
```

#### Supported Modules

```{toctree}
---
maxdepth: 1
---
llama_index_llm.md
```
