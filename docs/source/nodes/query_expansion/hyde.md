# HyDE

The HyDE is inspired by the paper "[Precise Zero-shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)". It uses a LLM model to generate a hypothetical passage for a given query.

## **Module Parameters**

**llm**: The query expansion node requires setting parameters related to the Large Language Model (LLM) being used. This includes specifying the LLM provider (e.g., `openai` or a list of providers like `[openai, huggingfacellm]` and the model configuration. By default, if only `openai` is specified without a model, the system uses the default model set in `llama_index`, which is `gpt-3.5-turbo`.
```{tip}
Information about the LLM model can be found [Supporting LLM models](../../local_model.md#supporting-llm-models).
```
- **Additional Parameters**: 
  - **batch**: How many calls to make at once. Default is 16.
  - Other LLM-related parameters such as `model`, `temperature`, and `max_token` can be set. These are passed as keyword arguments (`kwargs`) to the LLM object, allowing for further customization of the LLM's behavior.


## **Example config.yaml**
```yaml
modules:
- module_type: hyde
  llm: openai
  max_token: 64
```
