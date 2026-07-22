---
myst:
   html_meta:
      title: AutoRAG - RAG generation metrics
      description: Learn how to evaluate RAG generations (answers) in AutoRAG
      keywords: AutoRAG,RAG,RAG evaluation,RAG metrics,RAG metric,LLM metric
---
# Generation Metrics

## 1. Bleu

***BLEU*** (Bilingual Evaluation Understudy)

### 📌Definition

`n-gram` base

The extent to which words in the generated sentence are included in the reference sentence
→ By AutoRAG, the extent to which words in the `LLM generated result` are included in `Answer gt`

## 2. Rouge

***Rouge*** (Recall-Oriented Understudy for Gisting Evaluation)

### 📌Definition

`n-gram` base

The extent to which words from the reference sentences are included in the generated sentence
→ By AutoRAG, the extent to which words in `Answer gt` are included in the `LLM generated result`

## 3. METEOR

***METEOR*** (Metric for Evaluation of Translation with Explicit ORdering)

### 📌Definition

Here is the paper [link](https://www.cs.cmu.edu/~alavie/papers/BanerjeeLavie2005-final.pdf) that introduced ***METEOR***

The metric is based on the**harmonic mean**of unigram**precision and recall**, with recall weighted higher than
precision.

It also has several features that are not found in other metrics, such as stemming and synonym matching, along with the
standard exact word matching.

The metric was designed to fix some of the problems found in the more popular`BLEU`metric, and also produce good
correlation with human judgment at the sentence or segment level.

This differs from the `BLEU` metric in that `BLEU` seeks correlation at the corpus level.

## 4. Sem Score

### 📌Definition

Here is the paper [link](https://arxiv.org/pdf/2401.17072.pdf) that introduced ***Sem Score***.

The concept of SemScore is quite simple.

It measures semantic similarity between ground truth and the model’s generation using an embedding model.

You can find more detailed information
at [here](https://medium.com/@autorag/sem-score-maybe-the-answer-to-rag-evaluation-00db0d886d40)

## 5. G-Eval

### 📌 Definition

Here is the [link](https://arxiv.org/abs/2303.16634) that introduced ***G-Eval***

***G-Eval***, a framework of using large language models with **chain-of-thoughts** (CoT) and a form-filling paradigm,
to assess the quality of NLG outputs.

Paper said that **G-Eval with GPT-4** as the backbone model achieves a Spearman correlation of 0.514 with human on
a summarization task, outperforming all previous methods by a large margin.

So, in AutoRAG, we use **G-Eval with GPT-4**

---

### 5-1. Coherence

- Evaluate whether the answer is logically consistent and flows naturally.
- Evaluate the connections between sentences and how they fit into the overall context.

---

### 5-2. Consistency

- Evaluate whether the answer is consistent with and does not contradict the question asked or the information
  presented.
- An answer should provide information that does not conflict with the requirements of the question or the data
  presented.

---

### 5-3. Fluency

- Evaluate answers for fluency

---

### 5-4. Relevance

- Evaluate how well the answer meets the question's requirements
- A highly relevant answer should be directly related to the question's core topic or keyword.

### ❗How to use specific G-Eval metrics

You can use specific G-Eval metrics to use `metrics` parameter.

Here is an example YAML file that uses **G-Eval consistency** metric.

```yaml
- metric_name: g_eval
  metrics: [ consistency ]
```

## 6. Bert Score

### 📌Definition

Here is the [link](https://arxiv.org/pdf/1904.09675) that introduced ***BERT Score***.

A metric that measures the similarity between two sentences using BERT's Contextual Embedding.

Get the Contextual Embedding value of `Answer gt` and `LLM generated result` with BERT, evaluate the similarity with
Cosine Similarity for each token-pair, and weight each token with IDF.
