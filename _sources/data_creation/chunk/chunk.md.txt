# Chunk

In this section, we will cover how to chunk parsed result.

It is a crucial step because if the parsed result is not chunked well, the RAG will not be optimized well.

Using only YAML files, you can easily use the various chunk methods.
The chunked result is saved according to the data format used by AutoRAG.

## Overview

The sample chunk pipeline looks like this.

```python
from autorag.chunker import Chunker

chunker = Chunker.from_parquet(parsed_data_path="your/parsed/data/path")
chunker.start_chunking("your/path/to/chunk_config.yaml")
```

## Features

### 1. Add File Name
You need to set one of 'en'(=English), 'ko'(=Korean) or 'ja'(=Japanese)for the `add_file_name`parameter.
The 'add_file_name' feature is to add a file_name to chunked_contents.
This is used to prevent hallucination by retrieving contents from the wrong document.
Default form of English is `"file_name: {file_name}\n contents: {content}"`

#### Example YAML

```yaml
modules:
  - module_type: llama_index_chunk
    chunk_method: [ Token, Sentence ]
    chunk_size: [ 1024, 512 ]
    chunk_overlap: 24
    add_file_name: english
```

### 2. Sentence Splitter

The following chunk methods in the `llama_index_chunk` module use the sentence splitter.

- `Semantic_llama_index`
- `SemanticDoubling`
- `SentenceWindow`

The following methods use `PunktSentenceTokenizer` as the default sentence splitter.

See below for the available languages of `PunktSentenceTokenizer`.

["Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Italian, Malayalam, Norwegian, Polish, Portuguese, Russian, Slovenian, Spanish, Swedish, Turkish"]

So if the language you want to use is not in the list, or you want to use a different sentence splitter, you can use the sentence_splitter parameter.

#### Available Sentence Splitter
- [kiwi](https://github.com/bab2min/kiwipiepy) : For Korean 🇰🇷

#### Example YAML

```yaml
modules:
  - module_type: llama_index_chunk
    chunk_method: [ SentenceWindow ]
    sentence_splitter: kiwi
    window_size: 3
    add_file_name: english
```

#### Using sentence splitter that is not in the Available Sentence Splitter

If you want to use `kiwi`, you can use the following code.

```python
from autorag.data import sentence_splitter_modules, LazyInit

def split_by_sentence_kiwi() -> Callable[[str], List[str]]:
	from kiwipiepy import Kiwi

	kiwi = Kiwi()

	def split(text: str) -> List[str]:
		kiwi_result = kiwi.split_into_sents(text)
		sentences = list(map(lambda x: x.text, kiwi_result))

		return sentences

	return split

sentence_splitter_modules["kiwi"] = LazyInit(split_by_sentence_kiwi)
```

## Run Chunk Pipeline

### 1. Set chunker instance

```python
from autorag.chunker import Chunker

chunker = Chunker.from_parquet(parsed_data_path="your/parsed/data/path")
```

```{admonition} Want to specify project folder?
You can specify project directory with `--project_dir` option or project_dir parameter.
```

### 2. Set YAML file

Here is an example of how to use the `llama_index_chunk` module.

```yaml
modules:
  - module_type: llama_index_chunk
    chunk_method: [ Token, Sentence ]
    chunk_size: [ 1024, 512 ]
    chunk_overlap: 24
```

### 3. Start chunking

Use `start_chunking` function to start parsing.

```python
chunker.start_chunking("your/path/to/chunk_config.yaml")
```

### 4. Check the result

If you set `project_dir` parameter, you can check the result in the project directory.
If not, you can check the result in the current directory.

If the chunking is completed successfully, the following three types of files are created in the `project_dir`.

1. Chunked Result
2. Used YAML file
3. Summary file

For example, if chunking is performed using three chunk methods, the following files are created.
`0.parquet`, `1.parquet`, `2.parquet`, `parse_config.yaml`, `summary.csv`

Finally, in the summary.csv file, you can see information about the chunked result, such as what chunk method was used to chunk it.

## Output Columns
- `doc_id`: Document ID. The type is string.
- `contents`: The contents of the chunked data. The type is string.
- `path`: The path of the document. The type is string.
- `start_end_idx`:
  - Store index of chunked_str based on original_str before chunking
  - stored to map the retrieval_gt of Evaluation QA Dataset according to various chunk methods.
- `metadata`: It is also stored in the passage after the data of the parsed result is chunked. The type is dictionary.
  - Depending on the dataformat of AutoRAG's `Parsed Result`, metadata should have the following keys: `page`, `last_modified_datetime`, `path`.

#### Supported Chunk Modules

📌 You can check our all Chunk modules
at [here](https://edai.notion.site/Supporting-Chunk-Modules-8db803dba2ec4cd0a8789659106e86a3?pvs=4)

```{toctree}
---
maxdepth: 1
---
langchain_chunk.md
llama_index_chunk.md
```
