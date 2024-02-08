# 5. Prompt Maker

### 🔎 **Definition**
**Prompt Maker** is a module that makes a prompt from user query and retrieved contents.

## 🔢 **Parameters**
### **Overview**:
This document serves as a guide for configuring parameters, strategies, and the config YAML file for various nodes within a system.
### **Node Parameters**
- **None** 
### **Strategy**
**Performance Evaluation**: Like the query_expansion node, the Prompt Maker node's performance cannot be measured by the result alone. 
So, we evaluate the prompt, which is the result of the Prompt Maker node, using the generator (LLM). 
Therefore, the strategy sets the parameters needed to evaluate the answer of the generator (LLM).
```{tip}
Please refer to the parameter of [Generator Node](../generator/generator.md) for more details.
```

#### **Parameters**:

1. **Metrics**: Metrics such as `bleu`,`meteor`, and `rouge` are used to evaluate the performance of the prompt maker process through its impact on generator (llm) outcomes.
2. **Speed Threshold**: `speed_threshold` is applied across all nodes, ensuring that any method exceeding the average processing time for a query is not utilized.
3. **Generator Modules**: The prompt maker node can utilize all modules and module parameters from the generator node, including:
   - [llama_index_llm](../generator/llama_index_llm.md): with `llm` and additional llm parameters

### Example config.yaml file
```yaml
node_lines:
- node_line_name: post_retrieve_node_line  # Arbitrary node line name
  nodes:
    - node_type: prompt_maker
      strategy:
        metrics: [bleu, meteor, rouge]
        speed_threshold: 10
        generator_modules:
          - module_type: llama_index_llm
            llm: openai
            model: [gpt-3.5-turbo-16k, gpt-3.5-turbo-1106]
      modules:
        - module_type: fstring
          prompt: ["Tell me something about the question: {query} \n\n {retrieved_contents}",
                   "Question: {query} \n Something to read: {retrieved_contents} \n What's your answer?"]
```


#### Supported Modules

```{toctree}
---
maxdepth: 1
---
fstring.md
```