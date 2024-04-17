# Refine

The `Refine` module is generator based
on [llama_index](https://docs.llamaindex.ai/en/stable/examples/response_synthesizers/refine/).
Refine a response to a query across text chunks.

## **Module Parameters**

❗ **prompt**: Unlike the `llama_index_llm` and `vllm` modules, it runs the generator without using prompts created by
the `Prompt maker` node.
Default is `None`.

- Option 1️⃣ (recommend): If you don't set the prompt parameter, or set it to `None`, it uses the
  default `Refine prompt` from `llama index`.
- Option 2️⃣: The custom prompt used here is the template prompt for Refine, which does not need to
  include `{query}`, `{retrieved_contents}`.

**structured_answer_filtering**:
A [feature of llama index](https://docs.llamaindex.ai/en/stable/examples/response_synthesizers/structured_refine/) that,
when set to True, enables the feature.
Default is `False`

**LLM**: The refine module requires setting parameters related to the Large Language Model (LLM) being used.
This includes specifying the LLM provider (e.g., `openai` or a list of providers like `[openai, huggingfacellm]`) and
the model configuration.
By default, if only `openai` is specified without a model, the system uses the default model set in `llama_index`, which
is `gpt-3.5-turbo`.

```{tip}
Information about the LLM model can be found [Supporting LLM models](../../local_model.md#supporting-llm-models).
```

- **Additional Parameters**:
    - **batch**: How many calls to make at once. Default is 16.
    - Other LLM-related parameters such as `model`, `temperature`, and `max_tokens` can be set. These are passed as
      keyword arguments (`kwargs`) to the LLM object, allowing for further customization of the LLM's behavior.

## **Example config.yaml**

```yaml
        - module_type: refine
          llm: [ openai ]
          model: [ gpt-3.5-turbo-16k, gpt-3.5-turbo-1106 ]
          temperature: [ 0.5, 1.0, 1.5 ]
          structured_answer_filtering: [ True, False ]
          prompt: [ None, "Tell me something about the question and refine your answer:" ]
```