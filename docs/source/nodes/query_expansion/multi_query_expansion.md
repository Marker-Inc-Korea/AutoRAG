# Multi Query Expansion

The `multi_query_expansion` automates the process of prompt tuning by using an LLM to generate multiple queries from
different perspectives for a given user input query. The module uses a default multi-query prompt from
the [langchain MultiQueryRetriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/MultiQueryRetriever/)'
s default query prompt.

## **Module Parameters**

**llm**: The query expansion node requires setting parameters related to the Large Language Model (LLM) being used. This
includes specifying the LLM provider (e.g., `openai` or a list of providers like `[openai, huggingfacellm]`) and the
model configuration. By default, if only `openai` is specified without a model, the system uses the default model set
in `llama_index`, which is `gpt-3.5-turbo`.

```{tip}
Information about the LLM model can be found [Supporting LLM models](../../local_model.md#supporting-llm-models).
```

- **Additional Parameters**:
    - **batch**: How many llm calls to make at once. Default is 16.
    - Other LLM-related parameters such as `model`, `temperature`, and `max_token` can be set. These are passed as
      keyword arguments (`kwargs`) to the LLM object, allowing for further customization of the LLM's behavior. You can
      find these parameters at LlamaIndex docs.

## **Example config.yaml**

```yaml
modules:
- module_type: multi_query_expansion
  llm: openai
  temperature: [0.2, 1.0]
```
