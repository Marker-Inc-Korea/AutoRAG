---
myst:
  html_meta:
    title: AutoRAG - Multi Query Expansion
    description: Learn about multi query expansion module in AutoRAG
    keywords: AutoRAG,RAG,Advanced RAG,query expansion,multi query expansion
---
# Multi Query Expansion

The `multi_query_expansion` automates the process of prompt tuning by using an LLM to generate multiple queries from
different perspectives for a given user input query. The module uses a default multi-query prompt from
the [Langchain MultiQueryRetriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/MultiQueryRetriever/)
s default query prompt.

## **Module Parameters**

**llm**: The query expansion node requires setting parameters related to our generator modules.

- **generator_module_type**: The type of the generator module to use.
- **llm**: The type of llm.
- Other LLM-related parameters such as `model`, `temperature`, and `max_token` can be set. These are passed as keyword
  arguments (`kwargs`) to the LLM object, allowing for further customization of the LLM's behavior.

**Additional Parameters**:

- **prompt**: You can use your own custom prompt for the LLM model.
  Default prompt comes from langchain MultiQueryRetriever default query prompt.

## **Example config.yaml**

```yaml
modules:
- module_type: multi_query_expansion
  generator_module_type: llama_index_llm
  llm: openai
  model: [ gpt-3.5-turbo-16k, gpt-3.5-turbo-1106 ]
```
