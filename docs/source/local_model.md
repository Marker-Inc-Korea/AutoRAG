---
myst:
  html_meta:
    title: AutoRAG - Run local model in AutoRAG
    description: Learn how to run local model in AutoRAG
    keywords: AutoRAG,RAG,RAG model,RAG LLM,embedding model,local model
---

# Configure LLM & Embedding models

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

#### Using HuggingFace Models

There are two main ways to use HuggingFace models:

1. **Through OpenAILike Interface** (Recommended for hosted API endpoints):
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

2. **Through Direct HuggingFace Integration** (For local deployment):
```yaml
nodes:
  - node_line_name: node_line_1
    nodes:
      - node_type: generator
        modules:
          - module_type: llama_index_llm
            llm: huggingface
            model_name: mistralai/Mistral-7B-Instruct-v0.2
            device_map: "auto"
            model_kwargs:
              torch_dtype: "float16"
```

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

## Configure the Embedding model

### Modules that use Embedding model

Modules that using an embedding model can take `embedding_model` parameter to specify the LLM model.

- [vectordb](nodes/retrieval/vectordb.md)

### Supporting Embedding models

As default, we support OpenAI embedding models and some of the local models.
To change the embedding model, you can change the `embedding_model` parameter to the following values:

|                                           Embedding Model Type                                            |       embedding_model parameter       |
|:---------------------------------------------------------------------------------------------------------:|:-------------------------------------:|
|                             Default openai embedding (text-embedding-ada-002)                             |                openai                 |
|                              openai large embedding (text-embedding-3-large)                              |         openai_embed_3_large          |
|                              openai small embedding (text-embedding-3-small)                              |         openai_embed_3_small          |
|                  [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)                  |      huggingface_baai_bge_small       |
|               [cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2)               | huggingface_cointegrated_rubert_tiny2 |
| [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |     huggingface_all_mpnet_base_v2     |
|                             [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)                             |          huggingface_bge_m3           |

For example, if you want to use OpenAI text embedding large model, you can set `embedding_model` parameter
to `openai_embed_3_large` when setting vectordb.

```yaml
vectordb:
  - name: chroma_openai
    db_type: chroma
    client_type: persistent
    embedding_model: openai_embed_3_large
    collection_name: openai_embed_3_large
nodes:
  - node_line_name: node_line_1
    nodes:
      - node_type: retrieval
        modules:
          - module_type: vectordb
            vectordb: chroma_openai
```

### Add your embedding models

You can add more embedding models for AutoRAG.
You can add it by simply calling `autorag.embedding_models` and add new key and value.
For example,
if you want to add `[KoSimCSE](https://huggingface.co/BM-K/KoSimCSE-roberta-multitask)` model for Korean embedding,
execute the following code.

```python
import autorag
from autorag import LazyInit
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

autorag.embedding_models['kosimcse'] = LazyInit(HuggingFaceEmbedding, model_name="BM-K/KoSimCSE-roberta-multitask")
```

Then you can use `kosimcse` at config YAML file.

```{caution}
When you add new embedding model, you should use `LazyInit` class from autorag. The additional parameters have to be keyword parameter in the `LazyInit` initialization.
```

## Use vllm

You can use vllm to use local LLM. For more information, please check out [vllm](nodes/generator/vllm.md) generator
module docs.
