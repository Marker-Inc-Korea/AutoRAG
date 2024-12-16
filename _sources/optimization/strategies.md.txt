---
myst:
   html_meta:
      title: AutoRAG - Strategy
      description: Learn how to evaluate RAG generations (answers) in AutoRAG
      keywords: AutoRAG,RAG,RAG evaluation,RAG metrics,RAG metric,LLM metric
---
# Strategy

## Overview

From version 0.2.0 of AutoRAG, a new strategy option has been introduced to enhance the evaluation and selection of the
best module. Users can now choose between two methods: mean and rank. This document explains the new strategy parameter,
its options, and how to configure it.

## Strategy Parameter

The strategy parameter specifies the method used to evaluate and select the best module based on the defined metrics.
The options are:

- mean: The default method. It calculates the mean value of all specified metrics for each module and compares these
  mean values to determine the best module.

- rank: This method ranks each module's results per metric, calculates the reciprocal rank, and selects the best module
  based on these rank results.

- normalize mean: This method normalizes each metric value across modules to a common scale and then determines the
  best module.

## Configuration

To use the new strategy parameter, include it in the strategy section of your YAML configuration file.

### Example Configuration Using mean Strategy

```yaml
node_lines:
  - node_line_name: example_node_line_1
    nodes:
      - node_type: retrieval
        top_k: 10
        strategy:
          metrics: [ bleu, meteor, rouge ]
          speed_threshold: 10
          strategy: mean
```

### Example Configuration Using rank Strategy

```yaml
node_lines:
  - node_line_name: example_node_line_2
    nodes:
      - node_type: retrieval
        top_k: 5
        strategy:
          metrics: [ retrieval_precision, retrieval_recall ]
          speed_threshold: 5
          strategy: rank
```

### Example Configuration Using Normalize Mean Strategy

```yaml
node_lines:
  - node_line_name: example_node_line_2
    nodes:
      - node_type: retrieval
        top_k: 5
        strategy:
          metrics: [ retrieval_precision, retrieval_recall ]
          speed_threshold: 5
          strategy: normalize_mean
```

```{tip}
For more information, go to [custom config](./custom_config.md) and [optimization](./optimization.md) docs.
```
