# Parse

Using only YAML files, you can easily use the various document loaders of the langchain.
The parsed result is saved according to the data format used by AutoRAG.

## Output Columns

- texts: Parsed text from the document.
- path: Path of the document.
- pages: Number of pages in the document. Contains page if parsing on a per-page basis, otherwise -1.
    - Modules that parse per page: [ `clova`, `table_hybrid_parse` ]
    - Modules that don't parse on a per-page basis: [ `langchain_parse`, `llama_parse` ]
- last_modified_datetime: Last modified datetime of the document.

##

#### Supported Modules

```{toctree}
---
maxdepth: 1
---
langchain_parse.md
llama_parse.md
clova.md
table_hybrid_parse.md
```
