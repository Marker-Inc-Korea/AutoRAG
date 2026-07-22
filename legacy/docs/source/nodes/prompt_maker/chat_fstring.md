---
myst:
  html_meta:
    title: AutoRAG - Chat F-string
    description: Learn about Chat f-string module in AutoRAG
    keywords: AutoRAG,RAG,Advanced RAG,prompt,chat-template,chat
---

# Chat F-String

The `chat_fstring` module is prompt maker based on python’s f-string, but it follows the OpenAI chat template.
Make a chat prompt template using f-string from a query and retrieved_contents

## **Module Parameters**

**prompt**: This is the chat prompt that will be input to llm. Since it is created using a fstring, it must contain
`{query}` and `{retreived_contents}`.
Plus, you have to set "role" and "content" for each message.
You can put multiple messages as prompt.
Also, you can do experiment on multiple prompt templates by putting multiple messages in the list.

## **Example config.yaml**

```yaml
nodes:
  - node_type: prompt_maker
    modules:
      - module_type: chatfstring
        prompt:
          - - role: system
              content: "You are a helpful assistant that helps people find information."
            - role: user
              content: "Answer this question: {query}\n{retrieved_contents}"
          - - role: system
              content: "You are helpful."
            - role: user
              content: |
                Read the passages carefully and answer this question: {query}

                Passages: {retrieved_contents}
```
