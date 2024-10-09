# Query Generation

In this document, we will cover how to generate questions for the QA dataset.

## Overview

You can use `batch_apply` function at `QA` instance to generate questions.
Before generating a question, the `QA` must have the `qid` and `retrieval_gt` columns.
You can get those to use `sample` at the `Corpus` instance.

```{attention}
In OpenAI version of data creation, you can use only 'gpt-4o-2024-08-06' and 'gpt-4o-mini-2024-07-18'.
If you want to use another model, use llama_index version instead.
```

## Question types

1. [Factoid](#1-factoid)
2. [Concept Completion](#2-concept-completion)
3. [Two-hop Incremental](#3-two-hop-incremental)


## 1. Factoid
Factoid questions are those seeking brief, factual information that can be easily verified.
They typically require a yes or no answer or a brief explanation and often inquire about specific details such as dates, names, places, or events.

It supports "en" , "ko" or "ja" languages.

### Factoid Example

- What is the capital of France?
- Who invented the light bulb?
- When was Wikipedia founded?

### Usage
- OpenAI

```python
from openai import AsyncOpenAI
from autorag.data.qa.schema import QA
from autorag.data.qa.query.openai_gen_query import factoid_query_gen

qa = QA(qa_df)
result_qa = qa.batch_apply(factoid_query_gen, client=AsyncOpenAI(), lang="ko")
```

- LlamaIndex

```python
from llama_index.llms.openai import OpenAI
from autorag.data.qa.schema import QA
from autorag.data.qa.query.llama_gen_query import factoid_query_gen

llm = OpenAI()
qa = QA(qa_df)
result_qa = qa.batch_apply(factoid_query_gen, llm=llm, lang="ko")
```

## 2. Concept Completion
A “concept completion” question asks directly about the essence or identity of a concept.

It supports "en", "ko" or "ja" languages.

### Usage

- OpenAI

```python
from openai import AsyncOpenAI
from autorag.data.qa.schema import QA
from autorag.data.qa.query.openai_gen_query import concept_completion_query_gen

qa = QA(qa_df)
result_qa = qa.batch_apply(concept_completion_query_gen, client=AsyncOpenAI(), lang="ko")
```

- LlamaIndex

```python
from llama_index.llms.openai import OpenAI
from autorag.data.qa.schema import QA
from autorag.data.qa.query.llama_gen_query import concept_completion_query_gen

llm = OpenAI()
qa = QA(qa_df)
result_qa = qa.batch_apply(concept_completion_query_gen, llm=llm, lang="ko")
```

## 3. Two-hop Incremental

This query generation method is coming from [this paper](https://arxiv.org/pdf/2404.00571).
For making a robust multi-hop question, it first selects what will be the answer.
Then, it generates a question from the first document.
After that, it evolves a question from the second document to the multi-hop question.

We recommend you to use `openai` version, because it is more stable at the result. It uses structured output.

You can use "en" , "ko" or "ja" language.

### Example

- In which Mexican state can one find the Ciudad Deportiva, home to the Tecolotes de Nuevo Laredo?
- Which group has more members, New Jeans or Aespa?
- What is the name of the first album released by the band that performed at the 2022 Super Bowl halftime show?

### Usage

- OpenAI

```python
from openai import AsyncOpenAI
from autorag.data.qa.schema import QA
from autorag.data.qa.query.openai_gen_query import two_hop_incremental

qa = QA(qa_df)
result_qa = qa.batch_apply(two_hop_incremental, client=AsyncOpenAI())
```

- LlamaIndex

```python
from llama_index.llms.openai import OpenAI
from autorag.data.qa.schema import QA
from autorag.data.qa.query.llama_gen_query import two_hop_incremental

llm = OpenAI()
qa = QA(qa_df)
result_qa = qa.batch_apply(two_hop_incremental, llm=llm)
```
