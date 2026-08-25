# Legacy AutoRAG type checking

Legacy AutoRAG uses `ty` for static type checking. The checks are intentionally
introduced in small product-code scopes so optional integrations and third-party
stub limitations do not hide errors in the core modules.

`legacy/tests` is intentionally excluded from `ty`. Tests and test doubles are
validated with pytest and ruff instead; they are not part of the product-code
typing contract.

## Reproducible commands

From `legacy/`:

```bash
uv sync --extra ko --extra parse --extra ja
uv run --locked ty check autorag/data/chunk autorag/data/parse/base.py \
  autorag/data/parse/clova.py autorag/nodes/util.py
uv run --locked ruff check autorag tests
uv run --locked pytest -q tests/autorag/data/chunk \
  tests/autorag/data/parse/test_langchain_parse.py \
  tests/autorag/data/parse/test_clova.py
```

The focused product-code `ty` scope above is clean and runs in CI. Do not
replace it with `ty check autorag tests`: tests are intentionally excluded.
The complete product-code command, `uv run --locked ty check autorag`, remains
broader than this initial scope because it includes optional integrations and
third-party library APIs that are being migrated in subsequent issue-sized
changes.
