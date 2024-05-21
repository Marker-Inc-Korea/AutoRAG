# 5. Passage Filter

### 🔎 **Definition**

Passage filtering is a node that filters out passages.
It does not compress passages, but it deletes passages that are not relatable enough to the query.

```{admonition} What is difference between Passage Filter and Passage Reranker?
In passage reranker, you can set top_k parameter on your own.
Which means, reranker modules return 'top_k' passages you set all the time.

On the other hand, passage filter does not guarantee the number of passages to be returned.
It can be not filtered, or it can be filtered to 1 passage. 
```

### 🤸 **Benefits**

The primary benefit of passage filtering is that you can filter out irrelevant passages.
When the LLM gets irrelevant passages, it can be confused and return irrelevant answers.
So it is important to filter out irrelevant passages.

## **Node Parameters**

There are no node parameters for passage filter.

### Example config.yaml file

```yaml
node_lines:
  - node_line_name: retrieve_node_line  # Arbitrary node line name
    nodes:
      - node_type: passage_filter
        strategy:
          metrics: [ retrieval_f1, retrieval_recall, retrieval_precision ]
          speed_threshold: 5
        modules:
          - module_type: pass_passage_filter
          - module_type: similarity_threshold_cutoff
            threshold: 0.85
```

```{admonition} What is pass_passage_filter?
Its purpose is to test the performance that 'not using' any passage filter module.
Because it can be the better option that not using passage filter node.
So with this module, you can automatically test the performance without using any passage filter module.
```

```{toctree}
---
maxdepth: 1
---
similarity_threshold_cutoff.md
similarity_percentile_cutoff.md
recency_filter.md
threshold_cutoff.md
percentile_cutoff.md
```
