# Legacy AutoRAG contribution guide

## Type checking

`ty` is required for legacy product code only. Do not run `ty` against
`legacy/tests`, and do not add test files to the `ty` command or CI scope.

Tests are validated with pytest and ruff. Test doubles often intentionally
implement only the runtime surface needed by a test and are therefore outside
the legacy product-code typing contract.

The current clean product scope is documented in
`legacy/docs/source/ty.md`. Extend that scope with product modules as they are
made clean; keep tests excluded.
