# HuggingFace LLM x AutoRAG

## Using HuggingFace LLM with AutoRAG

For using HuggingFace LLM, you can use Llama Index LLm's `openailike` at the AutoRAG config YAML file without any further configuration.

### Writing the Config YAML File

Here’s the modified YAML configuration.

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

### Running AutoRAG

Before running AutoRAG, make sure you have your QA dataset and corpus dataset ready.
If you want to know how to make it, visit [here](../../data_creation/tutorial.md).

Run AutoRAG with the following command:

```bash
autorag evaluate \
 - qa_data_path ./path/to/qa.parquet \
 - corpus_data_path ./path/to/corpus.parquet \
 - project_dir ./path/to/project_dir \
 - config ./path/to/hf_config.yaml
```

AutoRAG will automatically experiment and optimize RAG.
