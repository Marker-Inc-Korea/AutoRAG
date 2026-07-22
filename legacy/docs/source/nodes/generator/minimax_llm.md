---
myst:
   html_meta:
      title: AutoRAG - MiniMax LLM
      description: Use MiniMax LLM in AutoRAG. Generate answers using MiniMax M2.7 and M2.5 models.
      keywords: AutoRAG,RAG,LLM,generator,MiniMax,M2.7,M2.5
---
# MiniMax LLM

The `minimax_llm` module integrates [MiniMax](https://www.minimax.io/) models into AutoRAG via the OpenAI-compatible API.

## Supported Models

| Model | Context Window |
|-------|---------------|
| MiniMax-M2.7 | 1,048,576 tokens |
| MiniMax-M2.7-highspeed | 1,048,576 tokens |
| MiniMax-M2.5 | 1,048,576 tokens |
| MiniMax-M2.5-highspeed | 204,800 tokens |

## Features

### Auto-truncate prompt

Prompts that exceed the model's token limit are automatically truncated to prevent API errors.

### Temperature clamping

MiniMax models accept temperature values between 0 and 1. Values above 1.0 are automatically clamped to 1.0.

### Think-tag stripping

MiniMax M2.5+ models may include `<think>...</think>` reasoning tags in their output. These are automatically stripped from the generated text.

## **Module Parameters**

- **llm**: The MiniMax model name. For example, `MiniMax-M2.7` or `MiniMax-M2.5-highspeed`.
- **batch**: The batch size for API calls. Default is 16.
- **truncate**: Whether to truncate input prompts to the model's max length. Default is True.
- **api_key**: MiniMax API key. You can also set this to env variable `MINIMAX_API_KEY`.
- And all parameters from the [OpenAI Chat Completion API](https://platform.openai.com/docs/api-reference/chat/create) (MiniMax uses an OpenAI-compatible endpoint).

## **Example config.yaml**

```yaml
modules:
  - module_type: minimax_llm
    llm: [MiniMax-M2.7, MiniMax-M2.5-highspeed]
    temperature: [0.1, 0.5]
    max_tokens: 512
    api_key: ${MINIMAX_API_KEY}
```
