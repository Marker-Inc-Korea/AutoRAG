# Data Creation

```{warning}
This is the beta version of new Data Creation.
This will be the main data creation pipeline at the AutoRAG v0.3 release.
At the time, the legacy version of data creation will be deprecated.

Plus, It is developing version. So there are some features that doesn't implemented yet.
And have potential bugs.
```

Data creation is the crucial process to use AutoRAG. Because AutoRAG needs an evaluation dataset for optimizing the RAG pipelines.
The following guide covers how to use LLM to create data in a form that AutoRAG can use.

## Basic Concepts

In this new data creation pipeline, we have three schemas. `Raw`, `QA`, and `Corpus`.

- `Raw`: Raw data after you parsed your documents. You can use this data to create `Corpus` data.
- `QA`: Question and Answer pairs. The main part of the dataset. You have to write a great question and answer pair for evaluating the RAG pipeline accurately.
- `Corpus`: The corpus is the text data that the LLM will use to generate the answer.
You can use the corpus to generate the answer for the question.
You have to make corpus data from your documents using parsing and chunking.

To see the tutorial of the data creation, check [here](tutorial.md).

```{toctree}
---
maxdepth: 2
---
tutorial.md
qa_creation/qa_creation.md
chunk/chunk.md
parse/parse.md
legacy/tutorial.md
legacy/ragas.md
```