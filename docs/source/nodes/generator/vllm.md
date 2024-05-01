# vllm

The `vllm` module is generator that using [vllm](https://blog.vllm.ai/2023/06/20/vllm.html).

## Why use vllm module?

`vllm` can generate new texts really fast. Its speed is more than 10x faster than a huggingface transformers library.

You can use `vllm` model with [llama_index_llm module](./llama_index_llm.md), but it is really slow because LlamaIndex
does not optimize for processing many prompts at once.

So, we decide to make a standalone module for vllm, for faster generation speed.

## **Module Parameters**

- **llm**: You can type your 'model name' at here. For example, `facebook/opt-125m`
  or `mistralai/Mistral-7B-Instruct-v0.2`.
- **max_tokens**: The maximum number of tokens to generate.
- **temperature**: The temperature of the sampling. Higher temperature means more randomness.
- **top_p**: Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1
  to consider all tokens.
- And all parameters
  from [LLM initialization](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L14)
  and [Sampling Params](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L25).

## **Example config.yaml**

```yaml
modules:
  - module_type: vllm
    llm: mistralai/Mistral-7B-Instruct-v0.2
    temperature: [ 0.1, 1.0 ]
    max_tokens: 512
```
