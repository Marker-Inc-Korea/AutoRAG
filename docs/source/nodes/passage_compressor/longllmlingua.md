# Long LLM Lingua

The `Long LLM Lingua` module is compressor based on [llmlingua](https://github.com/microsoft/LLMLingua).

Compresses the retrieved texts using LongLLMLingua.

## **Module Parameters**

**model_name**: The name of the LLM to be used for compression, defaulting to "NousResearch/Llama-2-7b-hf".

**instructions**: Optional instructions for the LLM, defaulting to "Given the context, please answer the final
question".

**target_token**: The target token count for the output, default to 300.

- **Additional Parameters**:
  You can put any additional parameters at llm_lingua.
  Find additional parameters [here](https://github.com/microsoft/LLMLingua)

## **Example config.yaml**

```yaml
modules:
  - module_type: longllmlingua
```
