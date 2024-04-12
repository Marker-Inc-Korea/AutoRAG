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

- [hyde](nodes/query_expansion/hyde.md)
- [query_decompose](nodes/query_expansion/query_decompose.md)
- [tree_summarize](nodes/passage_compressor/tree_summarize.md)
- [llama_index_llm](nodes/generator/llama_index_llm.md)

### Supporting LLM models

We support most of the llm that LlamaIndex is supporting.
To change the LLM model type, you can change the `llm` parameter to the following values:

|     LLM Model Type      |      llm parameter      |
|:-----------------------:|:-----------------------:|
|         OpenAI          |         openai          |
|     HuggingFaceLLM      |     huggingfacellm      |
|       OpenAILike        |       openailike        |


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
at [LlamaIndex LLM](https://docs.llamaindex.ai/en/latest/api_reference/llms.html).

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
from llama_index.llms.vllm import Vllm

autorag.generator_models['vllm'] = Vllm
```

Then you can use `mockllm` at config yaml file.

```{caution}
When you add new LLM model, you should add class itself, not the instance.

Plus, it must follow LlamaIndex LLM's interface.
```

## Configure the Embedding model

### Modules that use Embedding model

Modules that using embedding model can take `embedding_model` parameter to specify the LLM model.

- [vectordb](nodes/retrieval/vectordb.md)

### Supporting Embedding models

As default, we support OpenAI embedding models and some of the local models.
To change the embedding model, you can change the `embedding_model` parameter to the following values:

|                                           Embedding Model Type                                            |       embedding_model parameter       |
|:---------------------------------------------------------------------------------------------------------:|:-------------------------------------:|
|                                         Default openai embedding                                          |                openai                 |
|                                         openai babbage embedding                                          |            openai_babbage             |
|                                           openai ada embedding                                            |              openai_ada               |
|                                         openai davinci embedding                                          |            openai_davinci             |
|                                          openai curie embedding                                           |             openai_curie              |
|                  [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)                  |      huggingface_baai_bge_small       |
|               [cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2)               | huggingface_cointegrated_rubert_tiny2 |
| [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |     huggingface_all_mpnet_base_v2     |

For example, if you want to use OpenAI curie embedding model, you can set `embedding_model` parameter to `openai_curie`.

```yaml
nodes:
  - node_line_name: node_line_1
    nodes:
      - node_type: retrieval
        modules:
          - module_type: vectordb
            embedding_model: openai
```

```{attention}
You can't pass embedding model parameters at the config yaml file like LLM models.
Because the embedding model is initialized at the beginning of the AutoRAG program.
```

### Add your embedding models

You can add more embedding models for AutoRAG.
You can add it by simply calling `autorag.embedding_models` and add new key and value.
For example,
if you want to add `[KoSimCSE](https://huggingface.co/BM-K/KoSimCSE-roberta-multitask)` model for Korean embedding,
execute the following code.

```python
import autorag
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

autorag.generator_models['kosimcse'] = HuggingFaceEmbedding("BM-K/KoSimCSE-roberta-multitask")
```

Then you can use `kosimcse` at config yaml file.

```{caution}
When you add new LLM model, you should add instance of the `BaseEmbedding` class from LlamaIndex.
```
 
