# llama_index LLM

The `llama_index_llm` module is generator based on [llama_index](https://docs.llamaindex.ai/en/stable/api_reference/llms.html). It gets the LLM instance from llama index, and returns generated text by the input prompt. 
It does not generate log probs.

## **Module Parameters**

**llm**: The generator node requires setting parameters related to the Large Language Model (LLM) being used. This includes specifying the LLM provider (e.g., `openai` or a list of providers like `[openai, huggingfacellm]`) and the model configuration. By default, if only `openai` is specified without a model, the system uses the default model set in `llama_index`, which is `gpt-3.5-turbo`.
```{tip}
Information about the LLM model can be found [Supporting LLM models](../../local_model.md#supporting-llm-models).
```
- **Additional Parameters**: 
  - **batch**: How many calls to make at once. Default is 16.
  - Other LLM-related parameters such as `model`, `temperature`, and `max_token` can be set. These are passed as keyword arguments (`kwargs`) to the LLM object, allowing for further customization of the LLM's behavior.

## **Example config.yaml**
```yaml
modules:
- module_type: llama_index_llm
  llm: [openai]
  model: [gpt-3.5-turbo-16k, gpt-3.5-turbo-1106]
  temperature: [0.5, 1.0, 1.5]
```
