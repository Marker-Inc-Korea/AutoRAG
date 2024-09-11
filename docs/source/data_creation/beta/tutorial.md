# Start creating your own evaluation data

## Index

1. [Overview](#overview)
2. [Raw data to Corpus data](#make-corpus-data-from-raw-documents)
3. [Corpus data to QA data](#make-qa-data-from-corpus-data)
4. [Use custom prompt](#use-custom-prompt)
5. [Use multiple prompts](#use-multiple-prompts)
6. [If there are existing queries](#when-you-have-existing-qa-data)

## Overview
For the evaluation of RAGs we need data, but in most cases we have little or no satisfactory data.

However, since the advent of LLM, creating synthetic data has become one of the good solutions to this problem.

The following guide covers how to use LLM to create data in a form that AutoRAG can use.

---
![Data Creation](../../_static/data_creation.png)

AutoRAG aims to work with Python’s ‘primitive data types’ for scalability and convenience.

Therefore, to use AutoRAG, you need to convert your raw data into `corpus data`  and `qa data` to our [data format](./data_format.md).

## 1. Parse

Make parsed result from raw documents

available list 링크 갤러리

parse docs 링크 갤러리

YAML File Example

따로 만들고 싶으면 data format 맞추세유 ~


## 2. QA Creation

If you want to learn about more question generation type, check [here](./query_gen.md).


### Answer Creation

You can do like this.

```python
from autorag.data.beta.schema import QA
from autorag.data.beta.generation_gt.openai_gen_gt import make_basic_gen_gt
from openai import AsyncOpenAI

client = AsyncOpenAI()
qa = QA(qa_df)
result_qa = qa.batch_apply(make_basic_gen_gt, client=client)
```

Or using LlamaIndex

```python
from autorag.data.beta.schema import QA
from autorag.data.beta.generation_gt.llama_index_gen_gt import make_basic_gen_gt
from llama_index.llms.openai import OpenAI

llm = OpenAI()

qa = QA(qa_df)
result_qa = qa.batch_apply(make_basic_gen_gt, llm=llm)
```

## Chunking
