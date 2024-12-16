# QA creation

In this section, we will cover how to create QA data for the AutoRAG.

It is a crucial step to create the good QA data. Because if the QA data is bad, the RAG will not be optimized well.

## Overview

The sample QA creation pipeline looks like this.

```python
from llama_index.llms.openai import OpenAI

from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.generation_gt.llama_index_gen_gt import (
    make_basic_gen_gt,
    make_concise_gen_gt,
)
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.sample import random_single_hop

llm = OpenAI()
initial_corpus = initial_raw.chunk(
    "llama_index_chunk", chunk_method="token", chunk_size=128, chunk_overlap=5
)
initial_qa = (
    initial_corpus.sample(random_single_hop, n=3)
    .map(
        lambda df: df.reset_index(drop=True),
    )
    .make_retrieval_gt_contents()
    .batch_apply(
        factoid_query_gen,  # query generation
        llm=llm,
    )
    .batch_apply(
        make_basic_gen_gt,  # answer generation (basic)
        llm=llm,
    )
    .batch_apply(
        make_concise_gen_gt,  # answer generation (concise)
        llm=llm,
    )
    .filter(
        dontknow_filter_rule_based,  # filter don't know
        lang="en",
    )
)

initial_qa.to_parquet('./qa.parquet', './corpus.parquet')
```

### 1. Sample retrieval gt

To create question and answer, you have to sample retrieval gt from the corpus data.
You can get the initial chunk data from the raw data.
And then sample it using the `sample` function.

```python
from autorag.data.qa.sample import random_single_hop

qa = initial_corpus.sample(random_single_hop, n=3).map(
    lambda df: df.reset_index(drop=True),
)
```

You can change the sample method by changing the function to different functions.
Supported methods are below.

|      Method       |                Description                 |
|:-----------------:|:------------------------------------------:|
| random_single_hop |  Randomly sample one hop from the corpus   |
| range_single_hop  | Sample single hop with range in the corpus |


### 2. Get retrieval gt contents to generate questions

At the first step, you only sample retrieval gt ids. But to generate questions, you have to get the contents of the retrieval gt.
To achieve this, you can use the `make_retrieval_gt_contents` function.

```python
qa = qa.make_retrieval_gt_contents()
```

### 3. Generate queries

Now, you use LLM to generate queries.
In this example, we use the `factoid_query_gen` function to generate factoid questions.

```python
from llama_index.llms.openai import OpenAI

from autorag.data.qa.query.llama_gen_query import factoid_query_gen

llm = OpenAI()
qa = qa.batch_apply(
    factoid_query_gen,  # query generation
    llm=llm,
)
```

To know more query generation methods, check this [page](query_gen.md).

### 4. Generate answers

After generating questions, you have to generate answers (generation gt).

```python
from llama_index.llms.openai import OpenAI

from autorag.data.qa.generation_gt.llama_index_gen_gt import (
    make_basic_gen_gt,
    make_concise_gen_gt,
)

llm = OpenAI()

qa = qa.batch_apply(
    make_basic_gen_gt,  # answer generation (basic)
    llm=llm,
).batch_apply(
    make_concise_gen_gt,  # answer generation (concise)
    llm=llm,
)
```

To know more answer generation methods, check this [page](answer_gen.md).

### 5. Filtering questions

It is natural that LLM generates some bad questions.
So, it is better you filter some bad questions with classification models or LLM models.

To filtering, we use `filter` method.

```python
from llama_index.llms.openai import OpenAI

from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based

llm = OpenAI()
qa = qa.filter(
    dontknow_filter_rule_based,  # filter don't know
    lang="en",
)
```

To know more filtering methods, check this [page](filter.md).

### 6. Save the QA data

Now you can use the QA data for running AutoRAG.

```python
qa.to_parquet('./qa.parquet', './corpus.parquet')
```

```{toctree}
---
maxdepth: 1
---
query_gen.md
answer_gen.md
filter.md
evolve.md
```
