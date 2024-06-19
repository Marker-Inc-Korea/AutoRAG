# TroubleShooting

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
It is one of common mistakes to missing API key environment variable.
Use `echo` command to check your environment variable.

```bash
echo $OPENAI_API_KEY
```

If you don't see any output, you should set your API key environment variable.
```bash
export OPENAI_API_KEY=your_api_key
```

Often, in case you use `python-dotenv`, llm module can't recognize your environment variable.

2. Put your API key to config yaml file directly.
You can put your API key directly when you have difficulty adding environment variable.

Here is a sample config yaml file that uses api_key directly to generator module.
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
You can adjust batch size at our config yaml file. 
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

When the length of result is different from the original data, it is often caused by the index.

You must reset the index of your dataset before running AutoRAG.

```python
df = df.reset_index(drop=True)
```

## 3. LlamaIndex

### Facing Import Error

If you encountered the following llama_index ImportError, you need to check your LlamaIndex version.

If it is lower than 0.10.0, you need to use a version at least 0.10.0!

## 4. GPU-related Error

The error appears to be a VRAM out of memory error.

In this case, try lowering the `batch` (which can be set as a module parameter in YAML) as much as possible,
If that doesn't work, we recommend using a quantized model (if available)!

## 5. UnicodeDecodeError

Error reading a parquet file on Windows!

The workaround on Windows is to use engine='pyarrow',
This is something that needs to be fixed inside AutoRAG.

We’ll try to fix it in the [issue](https://github.com/Marker-Inc-Korea/AutoRAG/issues/494) :)

For now, please use Mac or Linux (or WSL on Windows)!
