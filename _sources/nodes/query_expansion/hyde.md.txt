---
myst:
  html_meta:
    title: AutoRAG - HyDE
    description: Learn about hyde module in AutoRAG
    keywords: AutoRAG,RAG,Advanced RAG,query expansion,HyDE
---
# HyDE

The HyDE is inspired by the paper "[Precise Zero-shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)".
It uses an LLM model to generate a hypothetical passage for a given query.

## **Module Parameters**

**llm**: The query expansion node requires setting parameters related to our generator modules.

- **generator_module_type**: The type of the generator module to use.
- **llm**: The type of llm.
- Other LLM-related parameters such as `model`, `temperature`, and `max_token` can be set. These are passed as keyword
  arguments (`kwargs`) to the LLM object, allowing for further customization of the LLM's behavior.

**Additional Parameters**:

- **prompt**: You can use your own custom prompt for the LLM model.
Default prompt is coming from the paper "[Precise Zero-shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)".

## **Example config.yaml**
```yaml
modules:
  - module_type: hyde
    generator_module_type: llama_index_llm
    llm: openai
    max_token: 64
```
