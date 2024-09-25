# Langchain Parse

Parse raw documents to use
[langchain document_loaders](https://python.langchain.com/v0.2/docs/integrations/document_loaders/#all-document-loaders).

## Available Parse Method by File Type

### 1. PDF

- [PDFMiner](https://python.langchain.com/v0.2/docs/integrations/document_loaders/pdfminer/)
- [PDFPlumber](https://python.langchain.com/v0.2/docs/integrations/document_loaders/pdfplumber/)
- [PyPDFium2](https://python.langchain.com/v0.2/docs/integrations/document_loaders/pypdfium2/)
- [PyPDF](https://python.langchain.com/v0.2/docs/integrations/document_loaders/pypdfloader/)
- [PyMuPDF](https://python.langchain.com/v0.2/docs/integrations/document_loaders/pymupdf/)
- [UnstructuredPDF](https://python.langchain.com/v0.2/docs/integrations/document_loaders/unstructured_pdfloader/)

#### Example YAML

```yaml
modules:
  - module_type: langchain_parse
    parse_method: [ pdfminer, pdfplumber ]
```

### 2. CSV

- [CSV](https://python.langchain.com/v0.2/docs/integrations/document_loaders/csv/)

#### Example YAML

```yaml
modules:
  - module_type: langchain_parse
    parse_method: csv
```

### 3. JSON

- [JSON](https://python.langchain.com/v0.2/docs/integrations/document_loaders/json/)

#### ❗Must have Parameter

- jq_schema: JSON Query schema to extract the content from the JSON file.

#### Example YAML

```yaml
  - module_type: langchain_parse
    parse_method: json
    jq_schema: .messages[].content
```

### 4. Markdown

- [UnstructuredMarkdown](https://python.langchain.com/v0.2/docs/integrations/document_loaders/unstructured_markdown/)

#### Example YAML

```yaml
  - module_type: langchain_parse
    parse_method: unstructuredmarkdown
```

### 5. HTML

- [BSHTML](https://python.langchain.com/v0.2/docs/integrations/document_loaders/bshtml/)

#### Example YAML

```yaml
  - module_type: langchain_parse
    parse_method: bshtml
```

### 6. XML

- [UnstructuredXML](https://python.langchain.com/v0.2/docs/integrations/document_loaders/xml/)

#### Example YAML

```yaml
  - module_type: langchain_parse
    parse_method: unstructuredxml
```

### 7. All files

- [Directory](https://python.langchain.com/v0.2/docs/how_to/document_loader_directory/)

#### 📌 API Needed

You need to have an API key to use the following document loaders.
- [Unstructured](https://python.langchain.com/v0.2/docs/integrations/document_loaders/unstructured_file/)
  - `UNSTRUCTURED_API_KEY` should be set in the environment variable.
- [UpstageLayoutAnalysis](https://python.langchain.com/v0.2/docs/integrations/document_loaders/upstage/)
  - `UPSTAGE_API_KEY` should be set in the environment variable.

#### Example YAML

```yaml
  - module_type: langchain_parse
    parse_method: upstagelayoutanalysis
```

## Using Parse Method that is not in the Available Parse Method

You can find more information about the document loaders at
[here](https://python.langchain.com/v0.2/docs/integrations/document_loaders/#all-document-loaders)

### How to Use

If you want to use `PyPDFDirectoryLoader` that is not in the available parse method, you can use the following code.

```python
from autorag.data import parse_modules
from langchain_community.document_loaders import PyPDFDirectoryLoader

parse_modules["pypdfdirectory"] = PyPDFDirectoryLoader
```

```{attention}
The key value in parse_modules must always be written in lowercase.
```
