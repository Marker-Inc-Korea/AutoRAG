---
name: autorag-setup
description: Configure AutoRAG for first use or repair its single-agent model, document roots, retrieval indexes, and health checks without exposing credentials.
---

# AutoRAG setup

Use this skill when AutoRAG is unconfigured, model resolution fails, or the user wants to prepare a document collection.

## Inspect safely

Check for `~/.autorag/config.json`, an explicit `--config` path, or `AUTORAG_CONFIG`. Never print credential values. The relevant config fields are:

- `searchPaths`
- `workspacePath`
- `memoryPath`
- `model.provider` and `model.id`
- `bm25`, `minSync`, `jikji`
- `datasources` and `datasourceAccess`

## Initialize

```bash
autorag init \
  --search-paths /path/to/documents \
  --workspace /path/to/workspace \
  --model-provider PROVIDER \
  --model-id MODEL
```

If the authenticated local runtime already supplies a usable model, the model flags may be omitted.

## Choose one search model

AutoRAG needs only one model. AutoRAG itself is the specialized librarian
agent: the configured model uses its retrieval tools, reads promising sources,
judges the evidence, and curates the answer in one loop.

For most collections, prefer a model with:

- reliable tool calling and structured-output behavior
- high output TPS and low first-token latency
- enough context for the expected source excerpts
- acceptable cost for several short search/read turns per query

A faster model usually makes interactive searches feel faster because one
AutoRAG search can involve several model turns around retrieval and source
reading. TPS does not make BM25, MinSync, Jikji, or filesystem indexing faster;
it reduces the model time between those tool calls. Use a larger or more
reasoning-heavy model only when the collection requires difficult synthesis,
conflict resolution, or specialized domain judgment.

Optional MinSync embedder flags:

```bash
autorag init \
  --embedder-id EMBEDDER \
  --embedder-base-url URL \
  --embedder-api-key-env ENV_NAME \
  --embedder-dimension N
```

Only store the environment variable name, never its secret value.

## Verify

```bash
autorag status
autorag health
autorag refresh
autorag search "summarize the collection" --top-k 3
```

- `status` checks corpus and index health without requiring a model.
- `health` resolves one model, verifies authentication, and optionally runs one completion probe.
- `refresh` updates parsed mirrors, BM25, MinSync, Jikji, and configured datasource indexes.
- `search` exercises the complete AutoRAG librarian workflow: retrieval,
  direct source reading, judgment, and structured curation with the configured
  model.

Use `--skip-probes` only when network access is intentionally unavailable; it leaves live provider completion unverified.

## Environment overrides

- `AUTORAG_CONFIG`
- `AUTORAG_SEARCH_PATHS`
- `AUTORAG_WORKSPACE`
- `AUTORAG_MEMORY_PATH`
- `AUTORAG_MODEL_PROVIDER`
- `AUTORAG_MODEL_ID`

Provider credentials stay in the provider's normal environment variable or authenticated runtime. Do not invent provider identities or model IDs.

## Completion condition

Setup is complete when folders are approved, `init` has written a non-secret config, `status` is acceptable, `health` resolves the model, `refresh` completes, and one real `autorag search` returns structured results.
