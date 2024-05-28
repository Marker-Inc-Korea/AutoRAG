# Query Decompose

The `query_decompose` is used to decompose a ‘multi-hop question’ into ‘multiple single-hop questions’ using a LLM model. The module uses a default decomposition prompt from the [Visconde paper](https://arxiv.org/pdf/2212.09656.pdf)'s StrategyQA few-shot prompt.

## **Module Parameters**

**llm**: The query expansion node requires setting parameters related to our generator modules.

- **generator_module_type**: The type of the generator module to use.
- **llm**: The type of llm.
- Other LLM-related parameters such as `model`, `temperature`, and `max_token` can be set. These are passed as keyword
  arguments (`kwargs`) to the LLM object, allowing for further customization of the LLM's behavior.

**Additional Parameters**:

- **prompt**: You can use your own custom prompt for the LLM model.
  default prompt comes from Visconde's StrategyQA few-shot prompt.

## **Example config.yaml**
```yaml
modules:
- module_type: query_decompose
  generator_module_type: llama_index_llm
  llm: openai
  model: [ gpt-3.5-turbo-16k, gpt-3.5-turbo-1106 ]
```
