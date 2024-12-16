# Langchain Chunk

Chunk parsed results to use [langchain text splitters](https://api.python.langchain.com/en/latest/text_splitters_api_reference.html#).

## Available Chunk Method

### 1. Token

- [SentenceTransformersToken](https://api.python.langchain.com/en/latest/sentence_transformers/langchain_text_splitters.sentence_transformers.SentenceTransformersTokenTextSplitter.html)

### 2. Character

- [RecursiveCharacter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)
- [character](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.CharacterTextSplitter.html)

### 3. Sentence

- [konlpy](https://api.python.langchain.com/en/latest/konlpy/langchain_text_splitters.konlpy.KonlpyTextSplitter.html): For Korean 🇰🇷

#### Example YAML

```yaml
modules:
  - module_type: langchain_chunk
    parse_method: konlpy
    add_file_name: korean
```

## Using Langchain Chunk Method that is not in the Available Chunk Method

You can find more information about the langchain chunk method at
[here](https://api.python.langchain.com/en/latest/text_splitters_api_reference.html#)

### How to Use

If you want to use `PythonCodeTextSplitter` that is not in the available chunk method, you can use the following code.

```python
from autorag.data import chunk_modules
from langchain.text_splitter import PythonCodeTextSplitter

chunk_modules["python"] = PythonCodeTextSplitter
```

```{attention}
The key value in chunk_modules must always be written in lowercase.
```
