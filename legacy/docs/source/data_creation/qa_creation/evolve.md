# Query Evolving

When you generate a query once, but you can evolve it in certain ways.
There are many ways to evolve your query to a new query.
It is common to get a naive, not specific, or realistic query at the first time.
So, you can evolve your query to a more specific, realistic, or better query.

There are so many ways to evolve, so it is hard to cover it all. So, we highly recommend you to write your own prompt for evolving.

Here, we provide some examples for query evolving with your custom promts.
After that, I will introduce query evolving functions.


## Make a Custom Evolving function

All you have to do is to write a function that calls LLM to evolving question with certain parameters.

We use `row` as a parameter, which is a row of a DataFrame with a Dict type.
So, you can use any value in the row of the QA dataframe.
And for using it on the `batch_apply` with async, you have to use `async def` for the function.

Here is an example of a custom-evolving function.

```python
from typing import Dict
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI

async def evolve_to_rude(row: Dict, llm: BaseLLM) -> Dict:
	original_query = row['query']
	system_prompt = ChatMessage(role=MessageRole.SYSTEM, content="Evolve the query to a rude question. Contain the original query's content, but make it super rude. Curse is okay to use.")
	user_prompt = ChatMessage(role=MessageRole.USER, content=f"Original question: {original_query}")
	response: ChatResponse = await llm.achat([system_prompt, user_prompt])
	row["query"] = response.message.content
	return row

llm = OpenAI()
qa = qa.batch_apply( # qa is a QA instance with "query" column
    evolve_to_rude,
    llm=llm,
)
```


## Provided Query Evolving Functions

You can replace the function to the provided functions and use pre-made evolving functions.


### 1. Reasoning Evolving

You can evolve the query to a reasoning question.
These questions are complicated, need complex reasoning, and hard to guess the answer.

This method is using RAGAS prompt.

- OpenAI : `from autorag.data.qa.evolve.openai_query_evolve import reasoning_evolve_ragas`
- LlamaIndex : `from autorag.data.qa.evolve.llama_index_query_evolve import reasoning_evolve_ragas`


### 2. Conditional Evolving

You can evolve the query to a conditional question.
It gives a condition and increases the complexity of the question.

This method is using RAGAS prompt.

- OpenAI : `from autorag.data.qa.evolve.openai_query_evolve import conditional_evolve_ragas`
- LlamaIndex : `from autorag.data.qa.evolve.llama_index_query_evolve import conditional_evolve_ragas`

### 3. Compress Query

This is a little bit different from other evolving functions. Usually, evolving functions make the question more complex.
But this function compresses the question to a simpler question.

Sometimes, the LLM generates a too complex or direct question with so much detailed information in the passages.
It can feel too much for a people. Because people really can't know too much information about their question.
So, you can compress the question to a simpler question with the same meaning. It can be effective to increase difficulties.

This method is using RAGAS prompt.

- OpenAI : `from autorag.data.qa.evolve.openai_query_evolve import compress_ragas`
- LlamaIndex : `from autorag.data.qa.evolve.llama_index_query_evolve import compress_ragas`
