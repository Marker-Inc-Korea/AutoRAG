# Migration Guide

1. [v0.3 migration guide](#v03-migration-guide)

# v0.3 migration guide

## Data Creation

From the v0.3 version, the previous data creation library goes into the `legacy` package.
Instead of legacy data creation, the `beta` package is introduced.
There are no longer `beta` package at the data, and you can use it without `beta` import.

For example,

- v0.2 version

```python
from autorag.data.corpus import langchain_documents_to_parquet
from autorag.data.qacreation import generate_qa_llama_index, make_single_content_qa
```

```python
from autorag.data.beta.query.llama_gen_query import factoid_query_gen
from autorag.data.beta.sample import random_single_hop
from autorag.data.beta.schema import Raw
```

- v0.3 version

```python
from autorag.data.legacy.corpus import langchain_documents_to_parquet
from autorag.data.legacy.qacreation import generate_qa_llama_index, make_single_content_qa
```

```python
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.sample import random_single_hop
from autorag.data.qa.schema import Raw
```
