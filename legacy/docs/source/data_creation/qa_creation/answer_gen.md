# Answer Generation

This is a generation for 'generation ground truth.'
It uses the LLM to generate the answer for the question and the given context (retrieval gt).

The answer generation methods can be used in AutoRAG is below.

1. [Basic Generation](#basic-generation)
2. [Concise Generation](#concise-generation)
3. [Custom Generation](#custom-generation)

## Basic Generation
This is just a basic generation for the answer.
It does not have specific constraints on how it generates the answer.

### OpenAI

```python
from autorag.data.qa.schema import QA
from autorag.data.qa.generation_gt.openai_gen_gt import make_basic_gen_gt
from openai import AsyncOpenAI

qa = QA(qa_df)
result_qa = qa.batch_apply(make_basic_gen_gt, client=AsyncOpenAI())
```

### LlamaIndex

```python
from autorag.data.qa.schema import QA
from autorag.data.qa.generation_gt.llama_index_gen_gt import make_basic_gen_gt
from llama_index.llms.openai import OpenAI

llm = OpenAI()
qa = QA(qa_df)
result_qa = qa.batch_apply(make_basic_gen_gt, llm=llm)
```

## Concise Generation
This is a concise generation for the answer.
Concise means that the answer is short and clear, just like a summary.
It is usually just a word that is the answer to the question.

### OpenAI

```python
from autorag.data.qa.schema import QA
from autorag.data.qa.generation_gt.openai_gen_gt import make_concise_gen_gt
from openai import AsyncOpenAI

qa = QA(qa_df)
result_qa = qa.batch_apply(make_concise_gen_gt, client=AsyncOpenAI())
```

### LlamaIndex

```python
from autorag.data.qa.schema import QA
from autorag.data.qa.generation_gt.llama_index_gen_gt import make_concise_gen_gt
from llama_index.llms.openai import OpenAI

llm = OpenAI()
qa = QA(qa_df)
result_qa = qa.batch_apply(make_concise_gen_gt, llm=llm)
```

## Custom Generation
You can generate answers with custom prompts. By adding a `system_prompt` of type `str`, you can generate answers in a different way than AutoRAG provides by default, or generate answers in languages other than "en", "ko", or "ja".
### LlamaIndex
```python
from autorag.data.qa.schema import QA
from autorag.data.qa.generation_gt.llama_index_gen_gt import make_custom_gen_gt
from llama_index.llms.openai import OpenAI

system_prompt = """As an expert AI assistant focused on providing accurate and concise responses, generate an answer for the question based strictly on the given **Text**.
Your answer should:
- Be derived only from the provided **Text** without using pre-trained knowledge.
- Contain only the answer itself, whether it is a full sentence or a single word, without any introductory phrases or extra commentary.
- If the information is unavailable within the **Text**, respond with "I don't know."
- be same as query's language"""
llm = OpenAI()
qa = QA(qa_df)
result_qa = qa.batch_apply(make_custom_gen_gt, llm=llm, system_prompt=system_prompt)
check_generation_gt(result_qa)
```
