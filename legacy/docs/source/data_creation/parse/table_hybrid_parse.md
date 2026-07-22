# Table Hybrid Parse

Parse raw documents using a combination of text and table parsing modules.

Because OCR models are paid models, it can be expensive to OCR-parse all raw documents.

OCR models are primarily used to parse raw documents that contain tables.
Therefore, it is cost-effective to parse raw documents that do not contain tables with non-OCR methods and parse raw documents that do contain tables with OCR.

To accomplish this, the Table Hybrid Parse module performs parsing in the following steps

1. breaks the raw document into pages.
2. uses table detection to distinguish between pages that contain tables and pages that do not.
3. pages that do not contain tables are parsed by the text parsing module.
4. Parses pages that contain tables with the table parsing module.
5. merge the parsing results to return the final result.

## Table Detection
Use `PDFPlumber` to split pages with and without tables.

## Table Parse Available Modules
- `llama_parse`
  - You need to add `result_type: markdown` to the table_params.
- `clova`
  - You need to add `table_detection: true` to the table_params.
- `langchain_parse`
  - You need to add `parse_method: upstagelayoutanalysis` to the table_params.

## Parameters
- `text_parse_module`: str
    - The module to use for text parsing.
- `text_params`: dict
  - parameters for the text parsing module
- `table_parse_module`: str
    - The module to use for table parsing.
- `table_params`: dict
  - parameters for the table parsing module

## Example YAML

If you want to use the `langchain_parse` module for text parsing and the `clova` module for table parsing, you can use the `table_hybrid_parse` module.

```yaml
modules:
  - module_type: table_hybrid_parse
    text_parse_module: langchain_parse
    text_params:
      parse_method: pdfplumber
    table_parse_module: clova
    table_params:
      table_detection: true
```
