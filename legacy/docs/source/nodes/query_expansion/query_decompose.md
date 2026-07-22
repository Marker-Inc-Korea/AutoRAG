---
myst:
   html_meta:
      title: AutoRAG - Query Decompose
      description: Learn about query decompose module in AutoRAG
      keywords: AutoRAG,RAG,Advanced RAG,query expansion,query decompose,visconde
---
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
Default prompt comes from StrategyQA few-shot prompt of Visconde.

## **Example config.yaml**
```yaml
modules:
- module_type: query_decompose
  generator_module_type: llama_index_llm
  llm: openai
  model: [ gpt-3.5-turbo-16k, gpt-3.5-turbo-1106 ]
```

## Default Prompt

When the question doesn't need decomposition, it must return "The question needs no decomposition."
Plus, each question will be allocated in `{question}`, so you have to write it in the prompt.

```
Decompose a question in self-contained sub-questions. Use \"The question needs no decomposition\" when no decomposition is needed.

Example 1:

Question: Is Hamlet more common on IMDB than Comedy of Errors?
Decompositions:
1: How many listings of Hamlet are there on IMDB?
2: How many listing of Comedy of Errors is there on IMDB?

Example 2:

Question: Are birds important to badminton?

Decompositions:
The question needs no decomposition

Example 3:

Question: Is it legal for a licensed child driving Mercedes-Benz to be employed in US?

Decompositions:
1: What is the minimum driving age in the US?
2: What is the minimum age for someone to be employed in the US?

Example 4:

Question: Are all cucumbers the same texture?

Decompositions:
The question needs no decomposition

Example 5:

Question: Hydrogen's atomic number squared exceeds number of Spice Girls?

Decompositions:
1: What is the atomic number of hydrogen?
2: How many Spice Girls are there?

Example 6:

Question: {question}

Decompositions:
```
