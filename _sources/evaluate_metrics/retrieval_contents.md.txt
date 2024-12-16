---
myst:
   html_meta:
      title: AutoRAG - RAG retrieval token metrics
      description: Learn how to evaluate passage compressor in AutoRAG
      keywords: AutoRAG,RAG,RAG evaluation,RAG metrics,RAG metric,passage compressor metric
---
# Retrieval Token Metrics

## 0. Retrieval token metric in AutoRAG

Currently, in AutoRAG, the ***Retrieval token metric*** is only used by the `Passage Compressor Node`. It measures
performance by comparing the compressed passage to Answer_gt.

When comparing Passage and Answer gt, the comparison is made on a per-token basis, which you can see by looking at the
example

### ✅Basic Example

answer gt = `['Do you want to buy some?']`

result = `['Do you want to buy some?', 'I want to buy some', 'I want to buy some water']`

First, let's break up gt and result into tokens

- GT is a total of 6 tokens
  `['do', 'you', 'want', 'to', 'buy', 'some']`
- The number of tokens in the result is 6, 5, and 6, respectively
  `['do', 'you', 'want', 'to', 'buy', 'some'],
  ['I', 'want', 'to', 'buy', 'some'],
  ['I', 'want', 'to', 'buy', 'some', 'water']`

Next, let's look at the number of overlapping tokens in gt and result

- The first is that all six tokens overlap with GT, so the number of overlapping tokens is 6.
- The second has four tokens overlapping except for the 'I.'
- The third has four tokens overlapping except for 'I' and 'water.'

## 1. Token Precision

### 📌Definition

Number of overlapping tokens / token length in result

### ✅Apply Basic Example

First, 6/6 = `1`

Second, 4/5 = `0.8`

Third, 4/6 = 2/3 = `0.666…`

Therefore, token precision is `0.822...`, the average of the three.

## 2. Token Recall

### 📌Definition

Number of overlapping tokens / token length in gt

### ✅Apply Basic Example

First, 6/6 = `1`

Second, 4/6 = `0.666…`

Third, 4/6 = 2/3 = `0.666…`

Therefore, Token Recall is `0.777…`, the average of three

## 3. Token F1

### 📌Definition

F1 score is the harmonic mean of **Precision** and **Recall**.

![f1_score](../_static/f1_score.png)

### ✅Apply Basic Example

Precision = `0.822…`

Recall = `0.777…`

Therefore, F1 Score = `0.797979…`
