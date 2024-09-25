---
myst:
  html_meta:
    title: AutoRAG RAGAS data gen documentation
    description: Generate RAG evaluation dataset using RAGAS
    keywords: AutoRAG,RAG,RAG evaluation,RAG dataset,RAGAS
---
# RAGAS evaluation data generation

RAGAS, the RAG evaluation framework, also support their advanced evaluation data generation.
You can learn more about their evaluation data generation method
at [here](https://docs.ragas.io/en/stable/concepts/testset_generation.html).

## Generate QA set from Corpus data using RAGAS

You can generate QA set from corpus data using RAGAS.

If you didn't make corpus data yet, you can make it by following the [tutorial](tutorial.md).

```python
import pandas as pd

from autorag.data.legacy.qacreation.ragas import generate_qa_ragas

corpus_df = pd.read_parquet('path/to/corpus.parquet')
qa_df = generate_qa_ragas(corpus_df, test_size=50)
```

This will make QA set with 50 questions using the RAGAS evaluation data generation method.

You can use output `qa_df` directly for AutoRAG optimization.

## RAGAS question types

- simple
- reasoning
- multi_context
- conditional

You can set the distribution of question types by making a distribution list.
You can access each question type by importing RAGAS evolution types.

```python
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from autorag.data.legacy.qacreation.ragas import generate_qa_ragas

distributions = {  # uniform distribution
    simple: 0.25,
    reasoning: 0.25,
    multi_context: 0.25,
    conditional: 0.25
}
qa_df = generate_qa_ragas(corpus_df, test_size=50, distributions=distributions)

```

Now you can pass this distribution dictionary as input.

## Use custom models

RAGAS supports custom models using Langchain.
You can get a support list of models from Langchain at [here](https://python.langchain.com/docs/integrations/llms/).

Plus, you might need to use `ChatModel` class from Langchain.

```python
from autorag.data.legacy.qacreation.ragas import generate_qa_ragas
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

generator_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
critic_llm = ChatOpenAI(model="gpt-4", temperature=0.0)
embedding_model = OpenAIEmbeddings()

qa_df = generate_qa_ragas(corpus_df, test_size=50, distributions=distributions,
                          generator_llm=generator_llm, critic_llm=critic_llm, embedding_model=embedding_model)
```
