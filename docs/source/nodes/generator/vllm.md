---
myst:
   html_meta:
      title: AutoRAG - vLLM
      description: Use vLLM in AutoRAG. Highly optimized for AutoRAG when you use local model on GPU.
      keywords: AutoRAG,RAG,LLM,generator,vLLM,LLM inference, AutoRAG multi gpu
---
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

## Support chat prompt

From v0.3.18, you can use chat prompt with `vllm` module.
For using chat prompt, you have to use `chat_fstring` module for prompt maker.

## Using reasoning

From v0.3.18, you can use reasoning with `vllm` module.
All you need to do is set `thinking` parameter to `True` in the YAML file.

```yaml
modules:
  - module_type: vllm
    llm: mistralai/Mistral-7B-Instruct-v0.2
    temperature: [ 0.1, 1.0 ]
    max_tokens: 512
    thinking: True
```

You have to use reasoning model to use reasoning, unless you can get an error.

## Use in Multi-GPU

First, for more details,
you must check out [vllm docs](https://docs.vllm.ai/en/latest/serving/distributed_serving.html) about parallel processing.

When you use multi gpu, you can set `tensor_parallel_size` parameter at YAML file.

```yaml
modules:
  - module_type: vllm
    llm: mistralai/Mistral-7B-Instruct-v0.2
    tensor_parallel_size: 2 # If the gpu is two.
    temperature: [ 0.1, 1.0 ]
    max_tokens: 512
```

Also, you can use any parameter from `vllm.LLM`, `SamplingParams`, and `EngineArgs`.

Plus, you can use it over v0.2.16, so you must be upgrade to the latest version.

```{warning}
We are developing multi-gpu compatibility for AutoRAG now.
So, please wait for the full compatibilty to multi-gpu environment.
```
```{warning}
When using the vllm module, errors may occur depending on the configuration of PyTorch. In such cases, please follow the instructions below:

1. Define the vllm module to operate in a single-case mode.
2. Set the skip_validation parameter to True when using the start_trial function in the evaluator.
```
