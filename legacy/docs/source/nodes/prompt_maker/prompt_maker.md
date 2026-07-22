---
myst:
   html_meta:
      title: AutoRAG - Prompt Maker
      description: Learn about prompt maker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,prompt
---
# 7. Prompt Maker

### 🔎 **Definition**
**Prompt Maker** is a module that makes a prompt from a user query and retrieved contents.

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

#### **Strategy Parameters**:

1. **Metrics**: (Essential) Metrics such as `bleu`,`meteor`, and `rouge` are used to evaluate the performance of the
   prompt maker process through its impact on generator (llm) outcomes.
2. **Speed Threshold**: (Optional) `speed_threshold` is applied across all nodes, ensuring that any method exceeding the
   average processing time for a query is not used.
3. **Token Threshold**: (Optional) `token_threshold` ensuring that output prompt average token length does not exceed
   the
   threshold.
4. **tokenizer**: (Optional) Since you don't know what LLM model you will use in the next nodes, you can specify the
   tokenizer name
   to use in `token_threshold` strategy.
   You can use OpenAI model names or Huggingface model names that support `AutoTokenizer`.
   It will automatically find the tokenizer for the model name you specify.
   Default is 'openai/gpt-oss-20b'
5. **Generator Modules**: (Optional, but recommended to set) The prompt maker node can use all modules and module
   parameters from the generator node,
   including:
   - [llama_index_llm](../generator/llama_index_llm.md): with `llm` and additional llm parameters
   - [openai_llm](../generator/openai_llm.md): with `llm` and additional openai `AsyncOpenAI` parameters
   - [vllm](../generator/vllm.md): with `llm` and additional vllm parameters

   And the default model of generator module for evaluating prompt maker is openai gpt-3.5-turbo model.

   Plus, the evaluation of the prompt maker will skip when there is only one combination of prompt maker modules and
   options.

### Example config.yaml file
```yaml
node_lines:
- node_line_name: post_retrieve_node_line  # Arbitrary node line name
  nodes:
    - node_type: prompt_maker
      strategy:
        metrics: [bleu, meteor, rouge, sem_score]
        speed_threshold: 10
        token_threshold: 1000
        tokenizer: gpt-3.5-turbo
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
chat_fstring.md
long_context_reorder.md
window_replacement.md
```
