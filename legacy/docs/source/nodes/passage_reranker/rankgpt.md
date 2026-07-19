---
myst:
   html_meta:
      title: AutoRAG - RankGPT Reranker
      description: Learn about rankGPT reranker module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,Reranker,RankGPT
---
# RankGPT

The 'RankGPT' module is a reranker using [RankGPT](https://github.com/sunnweiwei/RankGPT).
RankGPT uses GPT or other LLM model to reranking passages based on their relevance to a given query.

## **Module Parameters**

- `llm`: Default LLM model is gpt-3.5-turbo-16k, but you can use another LLM model to use RankGPT.
  The usage on the config YAML file for setting custom LLM is the same
  as [llama_index_llm](../generator/llama_index_llm.md) module.
  More details can be found on the [Local Model](../../local_model.md) page.
  Set `llm` as the name of LlamaIndex class name, and write the rest of parameters.
- `verbose`: The verbosity of the RankGPT module. Default is `False`.
- `batch`: Batch size. Since this module is using LLM model, choose wisely to prevent OOM or token limit error. Default is 16.
- `rankgpt_rerank_prompt`: The rerank prompt. Default is RankGPT default prompt.

## **Example config.yaml**

```yaml
modules:
  - module_type: rankgpt
    llm: openai
    model: gpt-4
    temperature: 0.5
    verbose: False
    batch: 8
```
