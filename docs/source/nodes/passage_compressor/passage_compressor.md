# 6. Passage_Compressor

### 🔎 **Definition**
Passage compressor module compresses the contents before they are input into a language model (LLM), reducing the amount of token usage.

### 🤸 **Benefits**
- **Efficiency in Token Usage:** By compressing the contents prior to their entry into a language model, the Passage Compressor significantly reduces the number of tokens required. This efficiency is crucial for large-scale applications where token usage can quickly accumulate, potentially leading to higher computational costs and slower processing times.
- **Cost Reduction:** Efficient token usage directly impacts the cost of running large language models. By reducing the number of tokens needed, the Passage Compressor can help lower operational costs associated with data processing and analysis.
- **Improved Performance:** Compressing contents before they are processed by a language model can lead to faster processing times. This is because the model has to deal with less data, which can speed up the analysis and generation process, making the system more responsive.


## 🔢 **Parameters**
### **Overview**:
This document serves as a guide for configuring parameters, strategies, and the config YAML file for various nodes within a system.
### **Node Parameters**
- **None** 
### **Strategy Parameters**
1. **Metrics**: The use of specialized metrics such as `retrieval_token_f1`, `retrieval_token_recall`, and `retrieval_token_precision` is crucial. These metrics are tailored to evaluate the efficiency of the passage compressor in optimizing the token-level relevance of retrieved content.
   ```{admonition} Purpose
   These metrics specifically address the need to calculate the F1 score, recall, and precision at the token level for evaluating the passage compressor's performance. This approach is distinct from the metrics used for evaluating other nodes like the retrieval node, query expansion node, and passage reranker node, underscoring the unique focus of the passage compressor on token-level content relevance.
   ```
2. Speed Threshold: An optional parameter that can be set to ensure the compressing process does not exceed a predefined processing time threshold.


### Example config.yaml file
```yaml
node_lines:
- node_line_name: retrieve_node_line  # Arbitrary node line name
  nodes:
  - node_type: passage_compressor
    strategy:
      metrics: [retrieval_token_f1, retrieval_token_recall, retrieval_token_precision]
      speed_threshold: 10
    modules:
      - module_type: pass_compressor
      - module_type: tree_summarize
        llm: openai
        model: gpt-3.5-turbo-16k
```

```{admonition} What is pass_compressor?
Its purpose is to test the performance that 'not using' any passage compressor module.
Because it can be the better option that not using passage compressor node.
So with this module, you can automatically test the performance without using any passage compressor module.
```


#### Supported Modules

```{toctree}
---
maxdepth: 1
---
tree_summarize.md
refine.md
longllmlingua.md
```
