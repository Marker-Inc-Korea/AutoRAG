# Evaluate your RAG
Did you optimize your RAG using AutoRAG?
You might want to compare your RAG and optimized RAG to see how much you improved.
You can evaluate your own RAG function easily using decorator from AutoRAG.
In other words, you can measure retrieval or generation performance on the RAG that you built already.

## Preparation

Before starting, ensure you have prepared a `qa.parquet` file for evaluation.
See [here](https://docs.auto-rag.com/data_creation/tutorial.html#qa-creation) for learning how to make QA dataset.

## Retrieval Evaluation

To compare the retrieval performance of your RAG with AutoRAG’s optimized version, follow these steps:

### `MetricInput` Dataclass

Start by building a `MetricInput` dataclass.
This structure includes several fields, but for retrieval evaluation, only query and retrieval_gt are mandatory.

Fields in MetricInput:

	1.	query: The original query.
	2.	queries: Expanded queries (optional).
	3.	retrieval_gt_contents: Ground truth passages (optional).
	4.	retrieved_contents: Retrieved passages (optional).
	5.	retrieval_gt: Ground truth passage IDs.
	6.	retrieved_ids: Retrieved passage IDs (optional).
	7.	prompt: The prompt used for RAG generation (optional).
	8.	generated_texts: Generated answers by the RAG system (optional).
	9.	generation_gt: Ground truth answers (optional).
	10.	generated_log_probs: Log probabilities of generated answers (optional).

### Using evaluate_retrieval

You can use the evaluate_retrieval decorator to measure performance. The decorator requires:

	1.	A list of metric_inputs.
	2.	The names of the metrics to evaluate.

Your custom retrieval function should return the following:

	1.	retrieved_contents: A list of retrieved passage contents.
	2.	retrieved_ids: A list of retrieved passage IDs.
	3.	retrieve_scores: A list of similarity scores.

### Important: Score Alignment

To ensure accurate performance comparisons, you need to adjust the similarity scores as follows:

|  Distance Metric  |         Adjusted Score          |
|:-----------------:|:-------------------------------:|
| Cosine Similarity | Use the Cosine Similarity value |
|    L2 Distance    |         1 - L2 Distance         |
|   Inner Product   |   Use the Inner Product value   |

Avoid using rank-aware metrics (e.g., mRR, NDCG, mAP) if you’re uncertain about the correctness of your similarity scores.

### Example Code
```python
import pandas as pd
from autorag.schema.metricinput import MetricInput
from autorag.evaluation import evaluate_retrieval

qa_df = pd.read_parquet("qa.parquet", engine="pyarrow")
metric_inputs = list(map(lambda x: MetricInput(
    query=x[1]["query"],
    retrieval_gt=x[1]["retrieval_gt"],
), qa_df.iterrows()))

@evaluate_retrieval(
    metric_inputs=metric_inputs,
    metrics=["retrieval_f1", "retrieval_recall", "retrieval_precision",
                   "retrieval_ndcg", "retrieval_map", "retrieval_mrr"]
)
def custom_retrieval(queries):
    # Your custom retrieval function
    # You have to return the retrieved_contents, retrieved_ids, retrieve_scores as List
    return retrieved_contents, retrieved_ids, retrieve_scores

retrieval_result_df = custom_retrieval(qa_df["query"].tolist())
```
Now you can see the result at the pandas DataFrame retrieval_result_df.

## Generation Evaluation

To evaluate the performance of RAG-generated answers, the process is similar to retrieval evaluation.

### `MetricInput` for Generation

For generation evaluation, the required fields are:

	•	query: The original query.
	•	generation_gt: Ground truth answers.

### Using evaluate_generation

The custom generation function must return:

	1.	generated_texts: A list of generated answers.
	2.	generated_tokens: A dummy list of tokens, matching the length of generated_texts.
	3.	generated_log_probs: A dummy list of log probabilities, matching the length of generated_texts.

Example Code

```python
import pandas as pd
from autorag.schema.metricinput import MetricInput
from autorag.evaluation import evaluate_generation

# Load QA dataset
qa_df = pd.read_parquet("qa.parquet", engine="pyarrow")

# Prepare MetricInput list
metric_inputs = [
    MetricInput(query=row["query"], generation_gt=row["generation_gt"])
    for _, row in qa_df.iterrows()
]

# Define custom generation function with decorator
@evaluate_generation(
    metric_inputs=metric_inputs,
    metrics=["bleu", "meteor", "rouge"]
)
def custom_generation(queries):
    # Implement your generation logic
    return generated_texts, [[1, 30]] * len(generated_texts), [[-1, -1.3]] * len(generated_texts)

# Evaluate generation performance
generation_result_df = custom_generation(qa_df["query"].tolist())
```

### Advanced Configuration

You can configure metrics using a dictionary. For example, if using semantic similarity (sem_score), specify additional parameters like the embedding model:

```python
@evaluate_generation(
    metric_inputs=metric_inputs,
    metrics=[
        {"metric_name": "sem_score", "embedding_model": "openai_embed_3_small"},
        {"metric_name": "bleu"}
    ]
)
```

By following these steps, you can effectively compare and evaluate your RAG system against the optimized AutoRAG pipeline.
