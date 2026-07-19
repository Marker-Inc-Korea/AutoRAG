# Llama Parse

Parse raw documents to use
[Llama Parse](https://github.com/run-llama/llama_parse).

## Set Environment Variables
You need to set the `LLAMA_CLOUD_API_KEY` environment variables to use Llama Parse.

You can get API Key at [here](https://docs.cloud.llamaindex.ai/llamaparse/getting_started/get_an_api_key)

## Language Support

You can find more information about the supported languages at
[here](https://github.com/run-llama/llama_parse/blob/main/llama_parse/utils.py#L16)

You can set language to use `language` parameter.

## Table Extraction

If you have tables in your raw document, set `result_type: markdown` to convert them to Markdown and save them.

📌`result_type`: You can set 3 types of result type.
- text
- markdown
- json

## Use Multimodal Model

You can see more information about multimodal model at [Multimodal Parsing](https://docs.cloud.llamaindex.ai/llamaparse/features/multimodal)

If you want to use multimodal model, you need to set the following parameters.

1. `use_vendor_multimodal_model`: Whether to use the vendor multimodal model. If you want to use multimodal model, set it to True. Default is False.
2. `vendor_multimodal_model_name`: The name of the vendor multimodal model. Default is "openai-gpt4o".
- You can find the list of available multimodal models [here](https://docs.cloud.llamaindex.ai/llamaparse/features/multimodal).

3. `use_own_key`: Whether to use the own API key. Default is False.
- If this is set to False, the [Basic Plan](https://docs.cloud.llamaindex.ai/llamaparse/features/multimodal) provided by llama parse will be used.
  - If set to False, only set environ variable `LLAMA_CLOUD_API_KEY` is required to use it.
- If true, you will need to set the api_key below.
- There are two ways to set up an API key.
  - Putting `vendor_multimodal_api_key` directly into the YAML File
  - Put the API Key in an environment variable based on `vendor_multimodal_model_name`.
    - `vendor_multimodal_model_name`: `openai-gpt4o` or `openai-gpt-4o-mini`
      - Set `OPENAI_API_KEY` environment variable
    - `vendor_multimodal_model_name`: `anthropic-sonnet-3.5`
      - Set `ANTHROPIC_API_KEY` environment variable
    - `vendor_multimodal_model_name`: `gemini-1.5-flash` or `gemini-1.5-pro`
      - Set the `GEMINI_API_KEY` environment variable


```{note}
vendor_multimodal_model_name: "custom-azure-model" is not supported in this module.
```


## Example YAML

- Not use multimodal model

```yaml
modules:
  - module_type: llama_parse
    result_type: markdown
    language: en
```

- Use multimodal model with [Basic Plan](https://docs.cloud.llamaindex.ai/llamaparse/features/multimodal) provided by llama parse
```yaml
modules:
  - module_type: llamaparse
    result_type: markdown
    use_vendor_multimodal_model: true
    vendor_multimodal_model_name: openai-gpt-4o-mini
```

- Use multimodal model with own API Key
```yaml
modules:
  - module_type: llamaparse
    result_type: markdown
    use_vendor_multimodal_model: true
    vendor_multimodal_model_name: openai-gpt-4o-mini
    use_own_key: true
    vendor_multimodal_api_key: YOUR_OPENAI_API_KEY
```

- Use multimodal model with own API Key (Environment Variable)
```yaml
modules:
  - module_type: llamaparse
    result_type: markdown
    use_vendor_multimodal_model: true
    vendor_multimodal_model_name: openai-gpt-4o-mini
    use_own_key: true
```
