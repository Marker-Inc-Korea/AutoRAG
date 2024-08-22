---
myst:
  html_meta:
    title: AutoRAG - Window Replacement
    description: Learn about Window Replacement module in AutoRAG
    keywords: AutoRAG,RAG,Advanced RAG,prompt
---

# Window Replacement

📌Only available for corpus chunked with `sentence window` method

The `window_replacement` module is prompt maker based on [llama_index](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo/).

Replace retrieved_contents with window to create a Prompt. This is most useful for large documents/indexes, as it helps
to retrieve more fine-grained details.

Make a prompt using `window_replacement` from a query and retrieved_contents.

## **Module Parameters**

**prompt**: This is the prompt that will be input to llm. Since it is created using an fstring, it must
contain `{query}` and `{retreived_contents}`.

## **Example config.yaml**

```yaml
modules:
  - module_type: window_replacement
    prompt: [ "Tell me something about the question: {query} \n\n {retrieved_contents}",
              "Question: {query} \n Something to read: {retrieved_contents} \n What's your answer?" ]
```
