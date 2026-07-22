---
myst:
  html_meta:
    title: AutoRAG - Run local model in AutoRAG
    description: Learn how to run local model in AutoRAG
    keywords: AutoRAG,RAG,RAG model,RAG LLM,embedding model,local model
---

# Configure LLM

## Index

- [Configure the LLM model](#configure-the-llm-model)
    - [Modules that use LLM model](#modules-that-use-llm-model)
    - [Supporting LLM models](#supporting-llm-models)
    - [Add more LLM models](#add-more-llm-models)
- [Configure the Embedding model](#configure-the-embedding-model)
    - [Modules that use Embedding model](#modules-that-use-embedding-model)
    - [Supporting Embedding models](#supporting-embedding-models)
    - [Add your embedding models](#add-your-embedding-models)

## Configure the LLM model

### Modules that use LLM model

Most of the modules that using LLM model can take `llm` parameter to specify the LLM model.

- [llama_index_llm](nodes/generator/llama_index_llm.md)

The following modules can use generator module, which including `llama_index_llm`.

- [hyde](nodes/query_expansion/hyde.md)
- [query_decompose](nodes/query_expansion/query_decompose.md)
- [multi_query_expansion](nodes/query_expansion/multi_query_expansion.md)
- [tree_summarize](nodes/passage_compressor/tree_summarize.md)
- [refine](nodes/passage_compressor/refine.md)

### Supporting LLM Models

We support most of the LLMs that LlamaIndex supports. You can use different types of LLM interfaces by configuring the `llm` parameter:

| LLM Model Type | llm parameter  | Description |
|:--------------:|:--------------:|-------------|
|     OpenAI     |     openai     | For OpenAI models (GPT-3.5, GPT-4) |
|   OpenAILike   |   openailike   | For models with OpenAI-compatible APIs (e.g., Mistral, Claude) |
|     Ollama     |     ollama     | For locally running Ollama models |
|    Bedrock     |    bedrock     | For AWS Bedrock models |

For example, if you want to use `OpenAILike` model, you can set `llm` parameter to `openailike`.

```yaml
nodes:
  - node_line_name: node_line_1
    nodes:
      - node_type: generator
        modules:
          - module_type: llama_index_llm
            llm: openailike
            model: mistralai/Mistral-7B-Instruct-v0.2
            api_base: your_api_base
            api_key: your_api_key
```

At the above example, you can see `model` parameter.
This is the parameter for the LLM model.
You can set the model parameter for LlamaIndex LLM initialization.
The most frequently used parameters are `model`, `max_token`, and `temperature`.
Please check what you can set for the model parameter
at [LlamaIndex LLM](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/).

#### Common Parameters

The most frequently used parameters for LLM configuration are:

- `model`: The model identifier or name
- `max_tokens`: Maximum number of tokens in the response
- `temperature`: Controls randomness in the output (0.0 to 1.0)
- `api_base`: API endpoint URL (for hosted models)
- `api_key`: Authentication key (if required)

For a complete list of available parameters, please refer to the
[LlamaIndex LLM documentation](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/).

### Add more LLM models

You can add more LLM models for AutoRAG.
You can add it by simply calling `autorag.generator_models` and add new key and value.
For example, if you want to add `MockLLM` model for testing, execute the following code.

```{attention}
It was major update for LlamaIndex to v0.10.0.
The integration of llms must be installed to different packages.
So, before add your model, you should find and install the right package for your model.
You can find the package at [here](https://pretty-sodium-5e0.notion.site/ce81b247649a44e4b6b35dfb24af28a6?v=53b3c2ced7bb4c9996b81b83c9f01139).
```

```python
import autorag
from llama_index.core.llms.mock import MockLLM

autorag.generator_models['mockllm'] = MockLLM
```

Then you can use `mockllm` at config YAML file.

```{caution}
When you add new LLM model, you should add class itself, not the instance.

Plus, it must follow LlamaIndex LLM's interface.
```


### Integration list

```{toctree}
---
maxdepth: 1
---
aws_bedrock.md
huggingface_llm.md
nvidia_nim.md
ollama.md
```
