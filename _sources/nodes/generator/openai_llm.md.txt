---
myst:
   html_meta:
      title: AutoRAG - OpenAI LLM
      description: Use OpenAI LLM in AutoRAG. This is optimized to use OpenAI model in AutoRAG.
      keywords: AutoRAG,RAG,LLM,generator,OpenAI,GPT,gpt-4,gpt-3.5
---
# OpenAI LLM

The `openai_llm` module is optimized openai llm module for AutoRAG.

## Why use `openai_llm` module?

There are several advantages using `openai` module in AutoRAG.

### 1. Auto-truncate prompt

Sometimes, prompt might exceed a token limitation of the model.
It will occur server-side error, and all your answer results will be gone.
To prevent this, `openai_llm` module truncate prompt to the max length of gpt model.

### 2. Accurate token output

In `llama_index_llm` module, it does not return proper tokens. It just return pseudo token using GPT2 tokenizer.

When you use `openai_llm` module, you can get real tokens that used in gpt model.
In the future, there will be a module that uses token for boosting RAG performance.

### 3. Accurate log prob output

In `llama_index_llm` module, it does not return proper log probs since llama index does not support it.

With `openai_llm` module, you can get real log probability to every token of generated answers.
In the future, there will be some modules that use log probability, like answer filter.

## Support chat prompt

From v0.3.19, you can use chat prompt with `openai_llm` module.
For using chat prompt, you have to use `chat_fstring` module for prompt maker.

## **Module Parameters**

- **llm**: You can type your 'model name' at here. For example, `gpt-4-turbo-2024-04-09` or `gpt-3.5-turbo-16k`
- **batch**: The batch size of openai api call. You should decrease when you got token limit error.
- **truncate**: Whether you truncate input prompts to model's max length. Default is True. Recommend you to keep this
  True.
- **api_key**: OpenAI API key. You can also set this to env variable `OPENAI_API_KEY`.
- And all parameters
  from [OpenAI Chat Completion](https://platform.openai.com/docs/api-reference/chat/create)
  without `n`, `logprobs`, `stream` and `top_logprobs`.

## **Example config.yaml**

```yaml
modules:
  - module_type: openai_llm
    llm: [ gpt-3.5-turbo, gpt-4-turbo-2024-04-09 ]
    temperature: [ 0.1, 1.0 ]
    max_tokens: 512
```
