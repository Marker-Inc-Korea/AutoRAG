# Make custom config yaml file

```{warning}
Read this documentation after you learned about the [structure](./folder_structure.md) of the AutoRAG. 
```

## Make yaml file

At first, you must make a new yaml file.
In your favorite editor, make new file and save it as `.yml` or `.yaml` extension.

## Make Node Line

The first thing you can do is to set the node lines.
Think of node line as a collection of nodes. 

You can make a node line like this.

```yaml
node_lines:
  - node_line_name: node_line_1
  - node_line_name: node_line_2
  - node_line_name: node_line_3
```

## Specify nodes

Now, you can set which nodes you want to use in each node line.
When you set nodes, you have to set **strategies** and **node parameters**.

Strategy is a method to evaluate each node. 
You can specify the method how to evaluate each node, like metrics, speed threshold, etc.

Node parameter is a parameter that set to all modules in the node. 
If you have a parameter that duplicates in all modules, you can set it to node parameter.

You can set multiple nodes in a node line. You can specify node_type to set which node you want to use.

```{tip}
Check out which node you can use, and what strategy and parameters they have in the [nodes](../nodes/index.md) section.
```

```yaml
node_lines:
  - node_line_name: node_line_1
    nodes:
      - node_type: retrieval
        top_k: 10
        strategy:
          metrics: [bleu, meteor, rouge]
          speed_threshold: 10
```

At the above example, you can see `top_k`. This is a node parameter for retrieval node.

And you can see `metrics` and `speed_threshold` at the strategy. You can set multiple metrics using a list.


## Specify modules

Lastly, you need to set modules for each node. 
You can set multiple modules for a single node.
Plus, you need to set module parameters for each module.

Most module parameters are optional, but it can be a hyperparameter that you want to optimize. 
You can set multiple module parameters using a list.
It will convert to combinations of module parameters, and AutoRAG will test it one by one.

```{seealso}
If you want to know how optimization works in AutoRAG, please check out [here](../optimization/optimization.md).
```

```{tip}
Check out which module you can use, and what parameters they have in the [nodes](../nodes/index.md) section.
```

Here is the example of setting modules.

```yaml
node_lines:
  - node_line_name: node_line_1
    nodes:
      - node_type: retrieval
        top_k: 10
        strategy:
          metrics: [bleu, meteor, rouge]
          speed_threshold: 10
        modules:
          - module_type: bm25
          - module_type: vectordb
            embedding_model: openai
          - module_type: hybrid_rrf
            target_modules: ('bm25', 'vectordb')
            rrf_k: [3, 5, 10]
```

```{admonition} What is tuple in the yaml file?
You can set tuple in the yaml file. As default, yaml file does not support tuple.
But for the AutoRAG, you can set tuple.
The list in config YAML file treat as a combination of parameters, but tuple treat as a single parameter.
In this way, the module like hybrid can get a collection of values as a single parameter.

Keep in mind, we use `ast.literal_eval` to parse the tuple in the AutoRAG.
So you have to write the tuple like you are writing a tuple in the python code.
```

## Use environment variable in the yaml file

It becomes easier to manage the api key or other secrets using environment variables.
You can use environment variables in the yaml file directly. 

```yaml
node_lines:
  - node_line_name: node_line_1
    nodes:
      - node_type: retrieval
        top_k: 10
        strategy:
          metrics: [sem_score]
        modules:
          - module_type: vectordb
            embedding_model: openai
            api_key: ${OPENAI_API_KEY} 
```

Look the `api_key` parameter. You can use environment variable directly in the yaml file.

Use '${}' to use environment variable in the yaml file.

```{tip}
If there is no environment variable, it returns a empty string. 
It can occur unintended action, so you have to set the environment variable before you run the AutoRAG.
```
