---
myst:
   html_meta:
      title: AutoRAG - RAG generation metrics
      description: Learn how to evaluate RAG generations (answers) in AutoRAG
      keywords: AutoRAG,RAG,RAG evaluation,RAG metrics,RAG metric,LLM metric,event loop AutoRAG
---
# TroubleShooting

## Frequently Asked Questions

### 1. Error when using AutoRAG on Jupyter Notebook or API server

If you face event loop-related issue while using Jupyter notebook, please run this before using AutoRAG.

```python3
import nest_asyncio
nest_asyncio.apply()
```

### 2. Corpus id not found in corpus_data.

When you face error like `ValueError: doc_id: 0eec7e3a-e1c0-4d33-8cc5-7e604b30339b not found in corpus_data.`
There will be several reasons for this error.

1. Check there is a passage augmenter on your YAML file.
   - The passage augmenter is not supporting a validation process now. But starting a trial runs validation process as default.
     So you need to disable running validation while starting a trial.

     ```python
      from autorag.evaluator import Evaluator

      evaluator = Evaluator(qa_data_path='your/path/to/qa.parquet', corpus_data_path='your/path/to/corpus.parquet',
                            project_dir='your/path/to/project_directory',)
      evaluator.start_trial('your/path/to/config.yaml', skip_validation=True)
     ```
     or
     ```bash
      autorag evaluate --config your/path/to/default_config.yaml --qa_data_path your/path/to/qa.parquet --corpus_data_path your/path/to/corpus.parquet --project_dir ./your/project/directory --skip_validation true
     ```

2. Delete the project directory or use another project directory

It might be you changed your corpus data, but don’t use the new project directory.
In AutoRAG, the project directory must be separated for each new corpus data or QA data.
Which means one dataset per one project directory is needed.

If you’re facing this error after you edit your corpus data, please use another project directory.

## 1. Installation

### Could not build wheels
If you face this kind of error during installation, try some methods below.

1. Upgrade pip version
    ```bash
    pip install --upgrade pip
    ```

2. Install gcc or c++ packages
    ```bash
    sudo apt-get install build-essential # linux
    brew install gcc # mac
    ```

If you still have trouble, please open an issue on [GitHub](https://github.com/Marker-Inc-Korea/AutoRAG/issues) or chat at our [Discord](https://discord.gg/P4DYXfmSAs) channel.

## 2. Optimization

### Facing OPENAI API error

1. Check your API key environment variable
It is one of the common mistakes to missing API key environment variable.
Use `echo` command to check your environment variable.

```bash
echo $OPENAI_API_KEY
```

If you don't see any output, you should set your API key environment variable.
```bash
export OPENAI_API_KEY=your_api_key
```

Often, in case you use `python-dotenv`, llm module can't recognize your environment variable.

2. Put your API key to config YAML file directly.
You can put your API key directly when you have difficulty adding environment variable.

Here is a sample config YAML file that uses api_key directly to the generator module.
This can be work because we put additional parameters to llm initialization,
which means you can put any parameters for LlamaIndex LLM model.
```yaml
    - node_type: generator
      strategy:
        metrics: [bleu, meteor, rouge]
      modules:
        - module_type: llama_index_llm
          llm: openai
          model: gpt-3.5-turbo
          batch: 4
          api_key: your_api_key
```

```{warning}
Commit and push config yaml file contains your API key can cause serious security problem.
```

```{tip}
Put api_key or api_base directly to your config yaml file sometimes useful.
When you using OpenAILike model (like VLLM openai server), you can put api_base and api_key to your config yaml file.
In this way, you can use both OpenAI model and custom model.
```

### Error while running LLM

It is common you face OOM (Out of Memory) error or out of rate limit error while running LLM.
In this case, we suggest you adjusting batch size.

1. Adjust batch size
You can adjust batch size at our config YAML file.
All modules that using LLM model can get `batch` as module parameter.

For example, using `batch` at `llama_index_llm` module:

```yaml
      modules:
        - module_type: llama_index_llm
          llm: openai
          model: [gpt-3.5-turbo-16k, gpt-3.5-turbo-1106]
          temperature: [0.5, 1.0, 1.5]
          batch: 4
```

See? You can put `batch` parameter to `llama_index_llm` module.

```{tip}
We recommend setting batch under 3 when you are using openai model.
In our experiment, it occurred rate limit error when the batch size was 4.
(Check out your tier and limit error at [here](https://platform.openai.com/account/limits).)
```

### The length or row is different from the original data

When the length of a result is different from the original data, it is often caused by the index.

You must reset the index of your dataset before running AutoRAG.

```python
df = df.reset_index(drop=True)
```

## 3. LlamaIndex

### Facing Import Error

If you encounter the following llama_index ImportError, you need to check your LlamaIndex version.

If it is lower than 0.11.0, you need to use a version at least 0.11.0!

## 4. GPU-related Error

The error appears to be a VRAM out-of-memory error.

In this case, try lowering the `batch` (which can be set as a module parameter in YAML) as much as possible,
If that doesn't work, we recommend using a quantized model (if available)!

## 5. Ollama `RequestTimeOut` Error

If you encounter `RequestTimeOut` error, you can adjust the `timeout` parameter in the `ollama` module.

```yaml
        modules:
          - module_type: llama_index_llm
            llm: ollama
            model: llama3
            request_timeout: 100  # ⇒ You can change the timeout value
```

If you increase the timeout value but doesn't resolve the error, it may be a ollama issue.
