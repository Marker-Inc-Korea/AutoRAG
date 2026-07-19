# Llama Index Chunk

Chunk parsed results to use [Llama Index Node_Parsers & Text Splitters](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/).

## Available Chunk Method

### 1. Token

- [Token](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/token_text_splitter/)

### 2. Sentence

- [Sentence](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/)

### 3. Window

- [SentenceWindow](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_window/)

### 4. Semantic

- [semantic_llama_index](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/semantic_splitter/)
- [SemanticDoubleMerging](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_double_merging_chunking/)

### 5. Simple

- [Simple](https://docs.llamaindex.ai/en/v0.10.19/api/llama_index.core.node_parser.SimpleFileNodeParser.html)

#### Example YAML

```yaml
modules:
  - module_type: llama_index_chunk
    chunk_method: [ Token, Sentence ]
    chunk_size: [ 1024, 512 ]
    chunk_overlap: 24
    add_file_name: english
```

## Using Llama Index Chunk Method that is not in the Available Chunk Method

You can find more information about the llama index chunk method at
[here](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/).

### How to Use

If you want to use `HTMLNodeParser` that is not in the available chunk method, you can use the following code.

```python
from autorag.data import chunk_modules
from llama_index.core.node_parser import HTMLNodeParser

chunk_modules["html"] = HTMLNodeParser
```

```{attention}
The key value in chunk_modules must always be written in lowercase.
```
