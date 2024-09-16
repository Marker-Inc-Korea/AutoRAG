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

### Example YAML

Here is an example of how to use the `langchain_parse` module.

```yaml
modules:
  - module_type: langchain_parse
    parse_method: [ pdfminer, pdfplumber ]
```

## Output Columns

- texts: Parsed text from the document.
- path: Path of the document.
- pages: Number of pages in the document. Contains page if parsing on a per-page basis, otherwise -1.
    - Modules that parse per page: [ `clova`, `table_hybrid_parse` ]
    - Modules that don't parse on a per-page basis: [ `langchain_parse`, `llama_parse` ]
- last_modified_datetime: Last modified datetime of the document.

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
