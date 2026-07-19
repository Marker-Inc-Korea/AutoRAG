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

## YAML File Setting Guide

### 1. Use All Files

Available parse modules are listed below.

- Langchain_parse (parse_method: directory)
- Langchain_parse (parse_method: unstructured)
- Langchain_parse (parse_method: upstagedocumentparse)
- Llama_parse
- Clova

Here is an example YAML file about full modules about `file_type: all_files`.

```yaml
modules:
  - module_type: langchain_parse
    file_type: all_files
    parse_method: [ directory, unstructured, upstagedocumentparse ]
  - module_type: clova
    file_type: all_files
    table_detection: true
  - module_type: llamaparse
    file_type: all_files
    result_type: markdown
    language: ko
    use_vendor_multimodal_model: true
    vendor_multimodal_model_name: openai-gpt-4o-mini
```

### 2. Use Specific Files

Six file types can have a direct parse method specified.
Only one parse_method can be specified for each file type.

If you are in the source document folder and do not specify a parse method, the Default Method is used for each file extension.
For example, if you have a csv file in a folder and you don't specify a parse_method, the csv file will be parsed as csv, which is the default method.

#### Default Parse Method
- PDF: pdfminer
- CSV: csv
- Markdown: unstructuredmarkdown
- HTML: bshtml
- XML: unstructuredxml

📌`JSON` does not default because you must specify `jq_schema` as the key value of the content.

Here is an example YAML file about full modules about specific file types.

```yaml
modules:
  # PDF
  - module_type: langchain_parse
    file_type: pdf
    parse_method: pdfminer
  # CSV
  - module_type: langchain_parse
    file_type: csv
    parse_method: csv
  # JSON
  - module_type: langchain_parse
    file_type: json
    parse_method: json
    jq_schema: .content
  # Markdown
  - module_type: langchain_parse
    file_type: md
    parse_method: unstructuredmarkdown
  # HTML
  - module_type: langchain_parse
    file_type: html
    parse_method: bshtml
  # XML
  - module_type: langchain_parse
    file_type: xml
    parse_method: unstructuredxml
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
    file_type: pdf
    parse_method: pdfminer
```

### 3. Start parsing

Use `start_parsing` function to start parsing.

```python
parser.start_parsing("your/path/to/parse_config.yaml")
```

### 4. Check the result

If you set `project_dir` parameter, you can check the result in the project directory.
If not, you can check the result in the current directory.

If the parsing is completed successfully, the following three types of files are created in the `project_dir`.

1. Parsed Result
2. Used YAML file
3. Summary file

#### Use all files

You can use only one parse method at a time.

Parsed result will be saved as `parsed_result.parquet`.

If you want to use two all_files parse method, you should run the parse pipeline twice with different two YAML files.

Finally, in the summary.csv file, you can see information about the parsed result, such as what parse method was used to parse it.

#### Use specific file types

For example, if the file types you want to parse are PDF, XML, and JSON,
you'll have `pdf.parquet`, `xml.parquet`, and `json.parquet` in your project dir.
And the result of concatenating all of them is `parsed_result.parquet`.

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
