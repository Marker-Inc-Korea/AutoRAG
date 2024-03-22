# Start creating your own evaluation data

## Index

1. [Overview](#overview)
2. [Raw data to Corpus data](#make-corpus-data-from-raw-documents)
3. [Corpus data to QA data](#make-qa-data-from-corpus-data)
4. [Use custom prompt](#use-custom-prompt)
5. [Use multiple prompts](#use-multiple-prompts)


## Overview
For the evaluation of RAGs we need data, but in most cases we have little or no satisfactory data.

However, since the advent of LLM, creating synthetic data has become one of the good solutions to this problem.

The following guide covers how to use LLM to create data in a form that AutoRAG can use.

---
![Data Creation](../_static/data_creation.png)

AutoRAG aims to work with Python’s ‘primitive data types’ for scalability and convenience.

Therefore, to use AutoRAG, you need to convert your raw data into `corpus data`  and `qa data` to our [data format](./data_format.md).


## Make corpus data from raw documents
1. Load your raw data to texts with loaders such as lama_index, LangChain, etc.
2. Chunk the texts into passages. Use Langchain, LlamaIndex, etc.
3. Make it into corpus data using util's datatype converter (based on llama_index).

```{tip}
The format for corpus data can be found [corpus data format](data_format.md#corpus-dataset)
```

## Make qa data from corpus data

```{tip}
The format for qa data can be found [qa data format](data_format.md#qa-dataset)
```

```python
from llama_index.llms.openai import OpenAI
from autorag.data.qacreation import generate_qa_llama_index

contents = ['content1', 'content2', 'content3']  # You can load your corpus contents to string list
llm = OpenAI(model='gpt-3.5-turbo', temperature=1.0)
result = generate_qa_llama_index(llm, contents, question_num_per_content=1)
```

`generate_qa_llama_index` is a function designed to generate **questions** and its **generation_gt** per content.
You can set the number of questions per content by changing `question_num_per_content` parameter.

```{admonition} What is passage?
Passage is chunked units from raw data.
```

## Use custom prompt

You can use custom prompt to generate qa data.
The prompt must contains two placeholders:

- {{text}}: The content string
- {{num_questions}}: The number of questions to generate

```python
from llama_index.llms.openai import OpenAI
from autorag.data.qacreation import generate_qa_llama_index

prompt = """
Generate question and answer pairs for the given passage.

Passage:
{{text}}

Number of questions to generate: {{num_questions}}

Example:
[Q]: What is this?
[A]: This is a sample question.

Result:
"""

contents = ['content1', 'content2', 'content3']  # You can load your corpus contents to string list
llm = OpenAI(model='gpt-3.5-turbo', temperature=1.0)
result = generate_qa_llama_index(llm, contents, prompt=prompt, question_num_per_content=1)
```

## Use multiple prompts

If you want to generate different types of question and answer pairs, you can use multiple prompts.
From now, we support distributing multiple prompts by randomly based on the ratio of each prompt.
It means that the prompt will be selected by ratio per passage.

For this, you must provide a dictionary.
The dictionary must have the key, which is the prompt text file path, and the value which is the ratio of the prompt.

```python 
from llama_index.llms.openai import OpenAI
from autorag.data.qacreation import generate_qa_llama_index_by_ratio

ratio_dict = {
    'prompt1.txt': 1,
    'prompt2.txt': 2,
    'prompt3.txt': 3
}

contents = [f'content{i}' for i in range(6)]  # You can load your corpus contents to string list
llm = OpenAI(model='gpt-3.5-turbo', temperature=1.0)
result = generate_qa_llama_index_by_ratio(llm, contents, ratio_dict, question_num_per_content=1, batch=6)
```

```{warning}
Remeber all prompts must have the placeholders `{{text}}` and `{{num_questions}}`.
```
