# Filtering

After generating QA dataset, you want to filter some generation results.
Because LLM is not perfect and has a lot of mistakes while generating datasets,
it is good if you use some filtering methods to remove some bad results.

The supported filtering methods are below.

1. [Rule-based Don't know Filter](#rule-based-dont-know-filter)
2. [LLM-based Don't know Filter](#llm-based-dont-know-filter)

# 1. Unanswerable question filtering

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
from autorag.data.beta.schema import QA
from autorag.data.beta.filter.dontknow import dontknow_filter_rule_based

qa = QA(qa_df, corpus)
filtered_qa = qa.filter(dontknow_filter_rule_based, lang="en").map(
		lambda df: df.reset_index(drop=True) # reset index
	)
```

You can use "en" and "ko" language.

### LLM-based Don't know Filter

We can use the LLM for filtering unanswerable questions.
It can classify the vague questions as unanswerable.
But since it uses the LLM, it is much slower and expensive than the rule-based don't know filter.

- OpenAI

```python
from openai import AsyncOpenAI
from autorag.data.beta.schema import QA
from autorag.data.beta.filter.dontknow import dontknow_filter_openai

openai_client = AsyncOpenAI()
qa = QA(qa_df, corpus)
filtered_qa = qa.batch_filter(dontknow_filter_openai, client=openai_client, lang="en").map(
        lambda df: df.reset_index(drop=True) # reset index
    )
```

- Llama Index

```python
from llama_index.llms.ollama import Ollama
from autorag.data.beta.schema import QA
from autorag.data.beta.filter.dontknow import dontknow_filter_llama_index

llm = Ollama(model="llama3")
qa = QA(qa_df, corpus)
filtered_qa = qa.batch_filter(dontknow_filter_llama_index, llm=llm, lang="en").map(
        lambda df: df.reset_index(drop=True) # reset index
    )
```
