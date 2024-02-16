# Dataset Format

There are two kinds of dataset, QA dataset and Corpus dataset.
You must follow a specified format for data input.

## QA Dataset

```{admonition} Long story short
You must have `qid`, `query`, `retrieval_gt`, and `generation_gt` columns at `qa.parquet` file.
```

QA dataset contains user's expected question and its ground truth answers.
Also, there will be relevant document ids as ground truth retrieval.
As you know, there can be lots of variations of *good answers* for a single question.

You must include the following columns at your `qa.parquet` file:

### qid

Unique identifier for queries. Its type is `string`.

```{warning}
Do not make a duplicate query id. It can occur unexpected behavior.
```

### query

The user's question. 
Its type is `string`.
In this data, you can imagine or collect user's input questions to your RAG system.
What kinds of questions users ask your data?

### retrieval_gt

```{admonition} Long story short
Save retrieval ground truth as 2d list. It can be 1d list or just string.
```

2D list of document ids. You can save retrieval ground truth ids as a list of lists.

Why do we need 2D list? Some metrics support 'and'/'or' conditions for retrieval.
Think about real-life questions.
To answer a single question, there can be more than one knowledge source.

For example, if I ask about `Which K-pop girl group has more members, New Jeans or Aespa?`,
you must look up documents about [New Jeans](https://en.wikipedia.org/wiki/NewJeans)
and [Aespa](https://en.wikipedia.org/wiki/Aespa) to answer the question.
This can be 'and' operation. You must look up two different documents to answer the question.

On the other hand, there will be lots of documents that contain 'New Jeans has five members' knowledge in the corpus.
Then, these documents can be 'or' operation.
You can answer the question with any of these documents.

So, in this case, the data will be like this.

```
[
 ['NewJeans1', 'NewJeans2'],
 ['Aespa1', 'Aespa2', 'Aespa3'],
]
```

It means you must look up one of 'NewJeans1,' 'NewJeans2' and one of 'Aespa1,' 'Aespa2,'
'Aespa3' to answer the question.

```{tip}
If you don't have enough information to build a full 2-d list of retrieval_gt, 
it is okay to save it as 1-d list or just string.
If you save it as 1-d list, it treats as 'and' operation.
```

This column is crucial because AutoRAG evaluate retrieval performance with this column.
It can affect hugely to optimization performance or nodes like retrieval, query expansion or passage reranker.

### generation_gt

```{admonition} Long story short
Save generation ground truth as a list. If you have a single gt answer, it can be just string.
```

A list of ground truth generation (answer) that you want to expect that LLM model generates.

There will be lots of variations of good answers for a single question.
So you can put all of them in a list.

```{tip}
If you have only one ground truth answer, you can save it as a string.
```


## Corpus Dataset

```{admonition} Long story short
You must have `doc_id`, `contents`, and `metadata` columns at `corpus.parquet` file.
```

You can make corpus dataset using your own documents.
In general, you have to load your documents to text, and chunk it to passages.
Once it is chunked, the single passage can be a row in the corpus dataset.
Your RAG system will retrieve passages in this corpus dataset, so chunking strategy is really important.

### doc_id
A unique identifier for each passage. Its type is `string`.

```{warning}
Do not make a duplicate doc id. It can occur unexpected behavior.
```

### contents

The actual contents. Type must be a `string`.

For saving contents, you must chunk your documents to passages.
There are lots of chunking strategies, and you can choose one of them.

```{note}
It will support multi-modal, like images, videos, etc. in the future.
But from an early version of AutoRAG, it only supports text. 

Plus, we have plans to support chunking optimization for your data.
```

### metadata

Metadata for your passages. 
It is a dictionary that contains information.

You must include `last_modified_datetime` key at metadata.
We recommend you to include modified datetime of your passages, but it is okay to put `datetime.now()` if you don't want to use time-related feature. 
The value of `last_modified_datetime` must be an instance of python `datetime.datetime` class.

```{tip}
If you don't have any metadata, you can put an empty dictionary.
It will create a default metadata for you. (like `last_modified_datetime` with `datetime.now()`)
```

```{attention}
Please check that you reset the indicies of your dataset files.
If not, it can occur unexpected behavior.
```

## Samples

Looking for an example of dataset? 
Check out our AutoRAG dataset collection at [huggingface](https://huggingface.co/collections/MarkrAI/autorag-evaluation-datasets-65c0ee87d673dcc686bd14b8). 

Or you can download it using a script in our GitHub repo.
Go [here](https://github.com/Marker-Inc-Korea/AutoRAG/tree/main/sample_dataset) and follow the instructions.
