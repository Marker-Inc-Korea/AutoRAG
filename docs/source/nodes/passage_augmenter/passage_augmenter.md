# Passage Augmenter

### 🔎 **Definition**

Passage augmenter is a node that augments passages.
As opposed to the passage filter node, this is a node that adds passages

### 🤸 **Benefits**

The primary benefit of passage augmenter is that allows users to fetch additional passages.

## **Node Parameters**

There are no node parameters for passage augmenter.

### Example config.yaml file

```yaml
node_lines:
  - node_line_name: retrieve_node_line  # Arbitrary node line name
    nodes:
      - node_type: passage_augmenter
        strategy:
          metrics: [ retrieval_f1, retrieval_recall, retrieval_precision ]
          speed_threshold: 5
        modules:
          - module_type: pass_passage_augmenter
          - module_type: prev_next_augmenter
            mode: next
```

```{admonition} What is pass_passage_augmenter?
Its purpose is to test the performance that 'not using' any passage augmenter module.
Because it can be the better option that not using passage augmenter node.
So with this module, you can automatically test the performance without using any passage augmenter module.
```

```{toctree}
---
maxdepth: 1
---
prev_next_augmenter.md
```
