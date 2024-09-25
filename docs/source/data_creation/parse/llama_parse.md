# Llama Parse

Parse raw documents to use
[Llama Parse](https://github.com/run-llama/llama_parse).

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

## Example YAML

```yaml
modules:
  - module_type: llama_parse
    result_type: markdown
    language: en
```
