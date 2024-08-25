---
myst:
   html_meta:
      title: AutoRAG - How optimization works?
      description: Learn how AutoRAG optimization actually works for optimal RAG pipeline
      keywords: AutoRAG,RAG,RAG optimize,RAG performance
---
# How optimization works

Understanding how optimization really works in AutoRAG can be great for writing your own configuration YAML file.
In this documentation, you can learn about how AutoRAG works under the hood.

## Swapping modules in Node

![Advanced RAG](https://github.com/Marker-Inc-Korea/AutoRAG/assets/96727832/79dda7ba-e9d8-4552-9e7b-6a5f9edc4c1a)

Here is the diagram of the overall AutoRAG pipeline.
Each node represents a node, and each node's result is passed to the next node.

```{admonition} Do I need to use all nodes?
No. The essential node for the 'working' RAG pipeline is `retrieval`, `prompt maker` and `generator`.

The other nodes are optional, so you can add it for the better performance.
```

But remember, you can set multiple modules and multiple parameters in each node.
And you get the best result among them.

To achieve this, AutoRAG first makes all possible combinations of modules and parameters in each node.
Then, it runs the pipeline with each combination and gets the result.
Finally, it selects the best result among them with the given strategies.

Let me show you an example.

```yaml
modules:
- module_type: llama_index_llm
  llm: [openai]
  model: [gpt-3.5-turbo-16k, gpt-3.5-turbo-1106]
  temperature: [0.5, 1.0, 1.5]
```

In this YAML file, there is a llama_index_llm module and parameters in it.
AutoRAG automatically generates combinations of each parameter.

```python
combinations = [
    {
        'module_type': 'llama_index_llm',
        'llm': 'openai',
        'model': 'gpt-3.5-turbo-16k',
        'temperature': 0.5
    }, {
        'module_type': 'llama_index_llm',
        'llm': 'openai',
        'model': 'gpt-3.5-turbo-16k',
        'temperature': 1.0
    }, {
        'module_type': 'llama_index_llm',
        'llm': 'openai',
        'model': 'gpt-3.5-turbo-16k',
        'temperature': 1.5
    }, {
        'module_type': 'llama_index_llm',
        'llm': 'openai',
        'model': 'gpt-3.5-turbo-1106',
        'temperature': 0.5
    }, {
        'module_type': 'llama_index_llm',
        'llm': 'openai',
        'model': 'gpt-3.5-turbo-1106',
        'temperature': 1.0
    }, {
        'module_type': 'llama_index_llm',
        'llm': 'openai',
        'model': 'gpt-3.5-turbo-1106',
        'temperature': 1.5
    }
]
```

See all the combinations?
Now, AutoRAG runs each combination of modules and parameters one by one.
You can see the results in the `summary.csv` file.

## Pass the best result to the next node

The best result from the previous node is passed to the next node.
It means that the next node uses the best result from the previous node as input.

In this way, each node does not have to know how the input result is generated.
Think of it as one of Markov Chain, which only needs the previous state to generate the next state.
The node itself does not need to know the whole pipeline or the previous states.

Why do we choose this method?
We know that the previous node result affects the next node result, and it does not always guarantee the best result.
For example, if the node choice was B and the next was A. But it can be possible that the best is A - A combination.

However, testing all combinations with all nodes can cause too many combinations and retries.
Our goal is to find the optimal pipeline, not 'the best pipeline' for the given time and resources.

## Evaluate Nodes that can't evaluate

Some nodes, like `query_expansion` or `prompt_maker` can't be evaluated directly.
If you want to evaluate these nodes directly, you have to build ground truth values for it.
Like 'ground truth of expanded query' or 'ground truth of prompt.'

And it is a hard job. You know, making retrieval gt and generation gt is still hard.

So we use evaluation of the next node.
In general, `retrieval` node comes after `query_expansion`.
At `query_expansion` strategy, you can specify the `retrieval` modules for evaluation.
In evaluation process, we retrieve the documents using the modules you specified,
and evaluate the `query_expansion` node with the retrieved documents.

Similar with `prompt_maker` and `generation` node.
We evaluate `prompt_maker` node with the `generation` node's result.

You might be wondering that this method is valid.
Think of this way.
In the node, we already find the best result among the combinations.
And its combination contain `the next node` setups.
In the next node, we will test all combinations of the next node.
So, its result will be at least the same as the best result from the previous node.
And probably, in most cases you can get a better result than the best result from the previous node.

## More optimization strategies

AutoRAG is an alpha version, and there are a lot of possible optimization methods we can develop in the future.
We are ready to develop more optimization strategies, and we are open to your suggestions or feedback.
