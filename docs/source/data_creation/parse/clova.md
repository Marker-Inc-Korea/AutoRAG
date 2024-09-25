# Clova

Parse raw documents to use Naver
[Clova OCR](https://guide.ncloud-docs.com/docs/clovaocr-overview).

Clova OCR divides the document into pages for parsing.

## Table Detection

If you have tables in your raw document, set `table_detection: true` to use clova ocr table detection feature.

### Point

#### 1. HTML Parser
Clova OCR provides parsed table information in complex JSON format.
It converts the complex JSON form of the table to HTML for storage in the LLM.

The parser was created by our own AutoRAG team and you can find the detailed code in the `json_to_html_table` function in `autorag.data.parse.clova`.

#### 2. The text information comes separately from the table information.
If your document is a table + text, the text information comes separately from the table information.
So when using table_detection, it will be saved in `{text}\n\ntable html:\n{table}` format.

## Example YAML

```yaml
modules:
  - module_type: clova
    table_detection: true
```
