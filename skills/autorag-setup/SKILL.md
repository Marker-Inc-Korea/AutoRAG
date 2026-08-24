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
- `search` exercises retrieval, direct source reading, judgment, and structured curation in one agent loop.

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
