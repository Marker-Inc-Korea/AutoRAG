# Parse

In this section, we will cover how to parse raw documents.

It is a crucial step to parse the raw documents.
Because if the raw documents are not parsed well, the RAG will not be optimized well.

Using only YAML files, you can easily use the various document loaders.
The parsed result is saved according to the data format used by AutoRAG.

## Overview

The sample parse pipeline looks like this.

```python
from autorag.parser import Parser

parser = Parser(data_path_glob="your/data/path/*")
parser.start_parsing("your/path/to/parse_config.yaml")
```

## Run Parse Pipeline

### 1. Set parser instance

```python
from autorag.parser import Parser

parser = Parser(data_path_glob="your/data/path/*")
```

#### 📌 Parameter: `data_path_glob`

Parser instance requires `data_path_glob` parameter.
This parameter is used to specify the path of the documents to be parsed.

Only glob patterns are supported.

You can use the wildcard character `*` to specify multiple files.

you can specify the file extension like `*.pdf` to specific file types.

```{admonition} Want to specify project folder?
You can specify project directory with `--project_dir` option or project_dir parameter.
```

### 2. Set YAML file

Here is an example of how to use the `langchain_parse` module.

```yaml
modules:
  - module_type: langchain_parse
    parse_method: [ pdfminer, pdfplumber ]
```

### 3. Start parsing

Use `start_parsing` function to start parsing.

```python
parser.start_parsing("your/path/to/parse_config.yaml")
```

### 4. Check the result

If you set `project_dir` parameter, you can check the result in the project directory.
If not, you can check the result in the current directory.

The way to check the result is the same as the `Evaluator` and `Chunker` in AutoRAG.

A `trial_folder` is created in `project_dir` first.

If the parsing is completed successfully, the following three types of files are created in the trial_folder.

1. Parsed Result
2. Used YAML file
3. Summary file

For example, if parsing is performed using three parse methods, the following files are created.
`0.parquet`, `1.parquet`, `2.parquet`, `parse_config.yaml`, `summary.csv`

Finally, in the summary.csv file, you can see information about the parsed result, such as what parse method was used to parse it.

## Output Columns

- `texts`: Parsed text from the document.
- `path`: Path of the document.
- `pages`: Number of pages in the document. Contains page if parsing on a per-page basis, otherwise -1.
    - Modules that parse per page: [ `clova`, `table_hybrid_parse` ]
    - Modules that don't parse on a per-page basis: [ `langchain_parse`, `llama_parse` ]
- `last_modified_datetime`: Last modified datetime of the document.

#### Supported Parse Modules

📌 You can check our all Parse modules
at [here](https://edai.notion.site/Supporting-Parse-Modules-e0b7579c7c0e4fb2963e408eeccddd75?pvs=4)


```{toctree}
---
maxdepth: 1
---
langchain_parse.md
llama_parse.md
clova.md
table_hybrid_parse.md
```
