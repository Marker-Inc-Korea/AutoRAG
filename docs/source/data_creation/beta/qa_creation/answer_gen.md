# Answer Generation

## Overview

## Usage

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
