# Filtering

After generating QA dataset, you want to filter some generation results.
Because LLM is not perfect and has a lot of mistakes while generating datasets,
it is good if you use some filtering methods to remove some bad results.

The supported filtering methods are below.

1. [Rule-based Don't know Filter](#rule-based-dont-know-filter)
2. [LLM-based Don't know Filter](#llm-based-dont-know-filter)

## 1. Unanswerable question filtering

Sometimes LLM generates unanswerable questions from the given passage.
If unintended unanswerable questions are generated, the retrieval optimization performance will be lower.
So, it is great to filter unanswerable questions after generation QA dataset.

## Don't know Filter

At the Don't know filter, we use generation_gt to classify the question is unanswerable or not.
If the question is unanswerable, the generation_gt will be 'Don't know.'

### Rule-based Don't know Filter

We can use the rule-based don't know filter to filter unanswerable questions.
This will just use the pre-made don't know sentences and filter the questions.

This is not perfect, but it is a simple and fast way to filter unanswerable questions.

```python
from autorag.data.qa.schema import QA
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based

qa = QA(qa_df, corpus)
filtered_qa = qa.filter(dontknow_filter_rule_based, lang="en").map(
    lambda df: df.reset_index(drop=True)  # reset index
)
```

You can use "en", "ko"  or "ja" language.

### LLM-based Don't know Filter

We can use the LLM for filtering unanswerable questions.
It can classify the vague questions as unanswerable.
But since it uses the LLM, it is much slower and expensive than the rule-based don't know filter.

- OpenAI

```python
from openai import AsyncOpenAI
from autorag.data.qa.schema import QA
from autorag.data.qa.filter.dontknow import dontknow_filter_openai

qa = QA(qa_df, corpus)
filtered_qa = qa.batch_filter(dontknow_filter_openai, client=AsyncOpenAI(), lang="en").map(
    lambda df: df.reset_index(drop=True)  # reset index
)
```

- Llama Index

```python
from llama_index.llms.ollama import Ollama
from autorag.data.qa.schema import QA
from autorag.data.qa.filter.dontknow import dontknow_filter_llama_index

llm = Ollama(model="llama3")
qa = QA(qa_df, corpus)
filtered_qa = qa.batch_filter(dontknow_filter_llama_index, llm=llm, lang="en").map(
    lambda df: df.reset_index(drop=True)  # reset index
)
```


## 2. Passage Dependent Filtering

Passage-dependent questions are those where the answer varies depending on the passage or context selected.
Even if you have the greatest retrieval system, the system will not find the exact passage from the passage-dependent questions.

Since the passage-dependent questions are almost impossible to get a ground truth passage,
it will decrease the discriminative power of evaluation dataset.

So, it is good to filter the passage-dependent questions after generating QA dataset.
We use LLM as the filtering model.

- OpenAI

```python
from openai import AsyncOpenAI
from autorag.data.qa.schema import QA
from autorag.data.qa.filter.passage_dependency import passage_dependency_filter_openai

en_qa = QA(en_qa_df)
result_en_qa = en_qa.batch_filter(
    passage_dependency_filter_openai, client=AsyncOpenAI(), lang="en"
).map(lambda df: df.reset_index(drop=True))
```

- LlamaIndex

```python
from autorag.data.qa.schema import QA
from llama_index.llms.openai import OpenAI
from autorag.data.qa.filter.passage_dependency import passage_dependency_filter_llama_index

llm = OpenAI(temperature=0, model="gpt-4o-mini")
en_qa = QA(en_qa_df)
result_en_qa = en_qa.batch_filter(
    passage_dependency_filter_llama_index, llm=llm, lang="en"
).map(lambda df: df.reset_index(drop=True))
```
