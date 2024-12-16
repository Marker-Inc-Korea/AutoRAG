---
myst:
   html_meta:
      title: AutoRAG generator (LLM) node documentation
      description: Configure LLM at the AutoRAG. Know the module types, evaluation metrics, and other strategies.
      keywords: AutoRAG,RAG,LLM,generator,LLM evaluation,RAG evaluation
---
# 8. Generator

### 🔎 **Definition**
A node that allows for experimentation with various Large Language Models (LLMs). This node is designed to facilitate the testing and evaluation of different LLMs to determine the most effective model for specific tasks or data sets.

## 🔢 **Parameters**

### **Overview**
This document serves as a guide for configuring parameters, strategies, and the config YAML file for various nodes within a system

### **Node Parameters**
- **None**

### **Strategy Parameters**
1. **Metrics**:
   - **Types**: `bleu`, `meteor`, `rouge`, `sem_score`, `g_eval`, `bert_score`
   ```{admonition} Purpose
   These metrics are used to evaluate the performance of language models by comparing model-generated text to ground truth texts.
   We are planning to add more metrics to evaluate generation performance.
   ```

   ```{admonition} sem_score
   Sem_score is a metric that evaluates the semantic similarity between ground truth and llm generation.
   It is quite simple, but effective to evaluate LLM systems.

   Since it uses embedding model, you can specify the embedding model name at config YAML file.
   Since AutoRAG v0.0.6, we support dictionary at strategy.
   You can check out this feature at the example config.yaml file below.
   ```

2. **Speed Threshold**:
   - **Description**: This optional parameter can be applied to all nodes to ensure that the processing time for a method does not exceed a predefined threshold.

3. **Token Threshold**:
   - **Description**: This optional parameter ensures that the average token length of the output (LLM's generations)
     does not exceed the threshold.

```{warning}
When you use `llama_index` module, the tokenizer will be gpt2 automatically.
Please refer to the [issue](https://github.com/Marker-Inc-Korea/AutoRAG/issues/259) for more details.
```

### Example config.yaml file
```yaml
- node_line_name: post_retrieve_node_line  # Arbitrary node line name
  nodes:
    - node_type: generator
      strategy:
        metrics:
           - metric_name: bleu
           - metric_name: meteor
           - metric_name: sem_score
             embedding_model: openai
           - metric_name: g_eval
             metrics: [consistency, fluency, relevance, coherence]
             model: gpt-4
           - metric_name: bert_score
             lang: en
             batch: 64
        speed_threshold: 10
        token_threshold: 4000
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
vllm.md
openai_llm.md
```
