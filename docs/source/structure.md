---
myst:
   html_meta:
      title: AutoRAG - AutoRAG structure
      description: Learn how AutoRAG works under the hood
      keywords: AutoRAG,RAG,RAG structure,RAG optimization,AutoRAG system
---
# Structure



## Explanation of concepts

### Node & Module

**`Node`: Node** is a higher-level concept that acts as a container for modules. It allows for the interchange of modules, meaning within a single node, multiple modules can be swapped or combined to find the optimal mix for processing data.
- **`Node Parameters`** are common parameters that apply across all modules within a node. They are set at the node level and affect the operation of each module contained within that node.


**`Module`: Module** refers to the individual components that can be fitted into a node. A system can have multiple modules within a node, and the number of modules can significantly increase over time, while the number of nodes remains limited.
- **`Module Parameters`** are specific to each module, allowing for the customization of individual module behavior within the broader context of the node they reside in.


```{tip}
If you want to know more about the node and module, please refer to the [Nodes & Modules](./nodes/index.md) section.
```

### Node Line

`Node Line`: A Collection of Nodes.

#### Example Node Lines
![Node Lines](./_static/node_lines.png)

- **Purpose and Future Enhancements**: Aims to support merging, splitting, and looping in node sequences. These functionalities are in development.
- **Modular RAG Integration**: Essential for Modular Retrieval-Augmented Generation, enhancing large language models by integrating retrieval mechanisms for dynamic data processing. Refer to documentation for more details.
- **Current Temporary Configuration**:Nodes are currently arranged temporarily to simulate Node Lines' intended functionalities until full integration is achieved.

```{tip}
Node lines can be changed at any time in the YAML file
```



### Strategy

`Strategy`: Strategy is what you decide to optimize ‘based on’ when optimizing. It is a crucial aspect of optimization that guides the selection and arrangement of optimization techniques.

- **Metrics:** It is essential for all nodes to include metrics as part of their optimization strategy. Metrics serve as a quantitative basis for evaluating the effectiveness of optimization efforts and guiding strategic decisions.
  ```{admonition} Summarize
    By default, every node must contain their own metric.
  ```
- **Speed Threshold**: Optionally, the `speed_threshold` parameter can be added to all nodes. This parameter serves as a criterion for optimization, focusing efforts on enhancing the speed of operations up to a specified threshold.
  ```{admonition} Summarize
   The speed threshold can be optionally applied to all nodes.
  ```
- **Node-Specific Strategies:** The optimization strategy may vary from one node to another within the same system. This flexibility allows for tailored optimization that addresses the unique requirements or limitations of each node.
  ```{admonition} Summarize
  1. Different metrics for different nodes.
  2. You can find more information in the [Nodes & Modules](./nodes/index.md) section.
  ```
