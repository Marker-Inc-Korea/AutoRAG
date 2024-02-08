# F-String

The `fstring` module is prompt maker based on python’s f-string. Make a prompt using f-string from a query and retrieved_contents.


## **Module Parameters**

**prompt**: This is the prompt that will be input to llm. Since it is created using an fstring, it must contain `{query}` and `{retreived_contents}`.

## **Example config.yaml**
```yaml
modules:
  - module_type: fstring
    prompt: ["Tell me something about the question: {query} \n\n {retrieved_contents}",
             "Question: {query} \n Something to read: {retrieved_contents} \n What's your answer?"]
```
