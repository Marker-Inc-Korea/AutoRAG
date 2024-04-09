# Long Context Reorder

[A study](https://arxiv.org/abs/2307.03172) observed that the best performance typically arises when crucial data is
positioned at the start or conclusion of the input context. Additionally, as the input context lengthens, performance
drops notably, even in models designed for long contexts.

Make a prompt using `long_context_reorder` from a query and retrieved_contents.

## **Module Parameters**

**prompt**: This is the prompt that will be input to llm. It must contain `{query}` and `{retreived_contents}`.

## **Example config.yaml**

```yaml
modules:
  - module_type: long_context_reorder
    prompt: [ "Tell me something about the question: {query} \n\n {retrieved_contents}",
              "Question: {query} \n Something to read: {retrieved_contents} \n What's your answer?" ]
```