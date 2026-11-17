---
title: "Retrieval methods"
description: "Which retrieval methods AutoRAG registers, how their results merge, dedupe, and get cut, and what the agent and MCP Lite each return."
---

# Retrieval methods

This page explains the retrieval methods AutoRAG registers, how the pipeline merges and cuts their results, how `scope` and a time window narrow a query, and what each surface (the librarian agent, the model-free CLI, and MCP Lite) receives.

## The method registry

Every retrieval source implements the [`RetrievalMethod`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/src/retrieval/types.ts) interface: `describe()` returns a descriptor and `retrieve(query, options)` returns `RetrievalResult[]` (`id`, `content`, `source`, `score`, `metadata`). `RetrievalMethodRegistry` (`src/retrieval/registry.ts`) holds them by registered name and rejects a duplicate name.

A descriptor carries:

| Field | Contents |
|---|---|
| `name` | Registered method name (unique) |
| `type` | `posix`, `vector`, `bm25`, `sparse`, `hybrid`, `visual`, or `remote` |
| `description` | Human-readable summary |
| `status` | `active` or `stub` |
| `capabilities` | Capability strings (`lexical`, `semantic`, `incremental`, `parsed-mirrors`, `virtual-paths`, `scoped`, …) |
| `datasourceId` | Set for a datasource-backed method; absent for a local method |
| `tags` | Optional descriptive tags |

`ParallelRetriever` runs every registered method in parallel. A method that throws is skipped and the healthy methods still answer, with one exception: a thrown `MinSyncRequiredError` fails the whole run, because MinSync is required and an answer that silently lacks content search is worse than an error.

### What AutoRAG registers

| Registered name | Type | Method | Notes |
|---|---|---|---|
| `hybrid` | `hybrid` (or `sparse` on a sparse-only index) | `MinSyncHybridMethod` | The MinSync fan-out method: SPLADE sparse plus dense vectors over shared CDC chunks, degrading to sparse by itself |
| `minsync` | `vector` (or `sparse` on a sparse-only index) | `MinSyncVectorMethod` | Owns MinSync sync/readiness and backs `semantic_search_local_docs`; deliberately **not** in the fan-out, so one index is not counted twice |
| `<datasource>-lexical` | `bm25` | Connector-backed datasource method | One method per configured datasource connection, with its `datasourceId` set; CLI-backed skills register their own |

Datasources that can serve a hybrid chain register one `FallbackRetrievalMethod` per connection (the hybrid when it works, a component only when it does not), so the pipeline sees a single method per datasource rather than double-counting a hybrid and its own component.

Jikji (`jikji_find`), file-name search (`everything_search`, `fsearch_search`), and web search (`web_search`, `web_fetch`) are agent tools, not registry methods. See [../indexing/overview.md](../indexing/overview.md) for indexing and [../datasources/overview.md](../datasources/overview.md) for datasource methods.

## Merging and deduplication

`ResultMerger` (`src/retrieval/merger.ts`) is the cross-method collator:

1. Each method's scores are min-max normalized to a comparable `0..1` band, so a BM25 score and a cosine score rank by position rather than raw magnitude.
2. Every distinct chunk is kept. A **pure duplicate** — the same `source` + `id` pair — is dropped; the first occurrence wins and later ones only record that the evidence was corroborated. Two different chunks of one document are not duplicates.
3. A source backed by several distinct chunks or methods gets a bounded corroboration bonus (at most `0.3`), enough to reorder ties without overpowering relevance.
4. Scores are rescaled to `0..1` again. Survivor metadata gains `retrievalMethods` (sorted), `sourceChunkCount`, and `duplicateHitCount` when the chunk was seen more than once.

`search_all_documents` is the exhaustive surface: it returns every distinct chunk the methods found, so a document whose several passages matched surfaces all of them, not one. A large context window holds the full pool; the merger withholds nothing.

## Record cut and limits

`RecordCut` (`src/retrieval/record-cut.ts`) stops one long document from taking every slot of a ranked list. A search asks the backend for more hits than the caller wants and keeps the caller's count spread across documents: the best half with at most `perRecord` passages from any one document, then the best passage of each document not yet shown, then rank order for whatever room is left. The cut is **on** unless `limits.recordCut.enabled` is `false`; tune it with:

| `limits.recordCut` field | Default | Range | Meaning |
|---|---|---|---|
| `enabled` | `true` | boolean | Set `false` to turn the record cut off |
| `depth` | `3` | 1-10 | The backend is asked for `depth` × the limit |
| `newRecordShare` | `0.5` | 0-1 | Share of the limit kept for documents not yet shown |
| `perRecordFloor` | `4` | 1-300 | A record keeps at least this many head slots |
| `perRecordDivisor` | `6` | 1-300 | A record keeps at most `max(perRecordFloor, limit / perRecordDivisor)` head slots |
| `resultChars` | `60000` | positive | Characters one search tool result may hold |
| `minReadChars` | `2000` | positive | Below this share a result is announced as not read |

The same module sizes what the model is shown: `planReadBudget` splits a result budget max-min fairly across the results of one tool call, so the first large result cannot spend all of it.

Other retrieval caps (`limits`, positive integers, omitted means the default) are summarized here and documented field by field in [../reference/configuration.md](../reference/configuration.md):

| `limits` field | Default | Meaning |
|---|---|---|
| `mergedEvidenceCeiling` | `500` | `search_all_documents` merge ceiling when the model omits `topK` |
| `singleDatasourceTopK` | `20` | Default `topK` for a single-datasource search when the model omits one |
| `minSyncTopK` | `50` | MinSync retrieval default `topK` |
| `minSyncScopedQueryTopK` | `100` | MinSync fetch cap when a `scope` narrows the query |
| `toolDescriptionInstanceScopes` | `8` | Instance scopes listed in one datasource tool description |
| `prefetch.jikjiTopK` | `30` with a reranker, else `10` | Jikji find candidate count for the first answer |
| `prefetch.minSyncTopK` | `100` with a reranker, else `10` | MinSync candidate count for the first answer |
| `prefetch.datasourceTopK` | `20` with a reranker, else `10` | Hits requested from each selected datasource |
| `prefetch.jikjiPathLimit` | `100` | Jikji answer paths handed to the verification prompt |
| `prefetch.sectionLimit` | `100` | Results rendered per baseline section |

## Scope and time window

Both narrowing arguments are optional and only narrow; neither can widen what a datasource's trusted config allows.

- **`scope`** is an opaque virtual-path scope such as `/docs` or `/docs/**`. It is applied after retrieval: MinSync matches it against each result's virtual path, and datasources that support source scope filter their results through `filterDatasourceScope`. A physical path outside the configured search roots is rejected with `invalid-retrieval-scope`; use a virtual scope instead. A scoped MinSync query over-fetches (`topK * 5`, `+20`, capped at `minSyncScopedQueryTopK`) so the filter has candidates to work with.
- **`since` / `until`** are an ISO-8601 time window `[since, until)`. A date-only value or a datetime without an offset is read in the host's local time zone; `Z` or `±HH:MM` is honored exactly; a calendar-invalid value such as `2026-02-31` is rejected. No store filters by time natively, so the window is a post-filter over an over-fetched candidate list: each method is asked for `max(5×topK, topK+20)`, capped at `300` candidates. A result's time is `metadata.publishedAt`, else the local file's `metadata.sourceMtimeMs`. A result with neither is excluded and reported, never passed through as if it were in range. A malformed bound throws the parse error verbatim; a window with `since >= until` is rejected as empty.

When a window is applied, `retrieveWithDiagnostics` returns a `window` report (`examined`, `matched`, `untimed`, `ceilingHit`, `fetchDepth`) so an empty answer can say why it is empty.

## Diagnostics and unsearched surfaces

The pipeline returns diagnostics alongside partial results, and every surface that was not searched for the query appears in `unsearched` with the underlying error **verbatim** (CLI stderr, exit status, paths included):

| Code | Severity | Meaning |
|---|---|---|
| `retrieval-method-failed` | warning | A method threw and was skipped, or a method answered through a fallback after its primary failed |
| `retrieval-source-missing` | warning | Hits whose source file no longer exists were dropped; refresh the index to drop the stale entries |
| `rerank-failed` | warning | The reranker failed; the merged order is preserved (see [reranking.md](reranking.md)) |
| `time-window-untimed` | warning | A result was excluded from the time window because it has no `publishedAt` and no source mtime |
| `time-window-ceiling` | info | A method returned as many candidates as the over-fetch depth, so matches inside the range may be unexamined |

`unsearched` collapses per-method failures into one entry per surface: local MinSync methods share the `minsync` surface, a datasource method keeps its datasource id, and anything else reports under its own method name.

## What each surface receives

The librarian agent gets the merged pool with evidence ids:

- `search_all_documents` takes `{ query, topK?, scope?, since?, until? }` and returns every distinct chunk from every registered method, with an `[eN]` evidence id beside each result. Extra model-provided properties are never forwarded.
- `search_datasource_<id>` searches one connection only and is **not** model-reranked; its merged order is kept.
- `semantic_search_local_docs` runs MinSync vector retrieval over the parsed mirrors.
- `jikji_find`, `everything_search` (Windows), `fsearch_search` (macOS/Linux) locate files by name; `web_search` / `web_fetch` reach the public internet.

There is no `lexical_search_local_docs` tool: BM25-style lexical search runs inside the datasource chunk store and inside retrieval memory's similar-question search, and MinSync's lexical leg is SPLADE. Every retrieval tool prints `[eN]` ids; see [evidence-and-citations.md](evidence-and-citations.md).

The **model-free** surfaces (the `autorag lite retrieve` CLI and MCP Lite) call the same pipeline without the librarian:

- `autorag lite retrieve` retrieves and prints results without model curation:

```bash
autorag lite retrieve "renewal clause" --top-k 20 --scope /contracts
autorag lite retrieve "quarterly numbers" --since 2026-01-01 --until 2026-04-01 --strict
```

  Its flags are `--top-k`, `--scope`, `--since`, `--until`, `--refresh` (rebuild the indexes incrementally first), and `--strict` (fail instead of answering from a stale index).
- MCP `autorag.search` takes `{ query, topK?, scope?, since?, until?, strict?, datasourceIds?, methods?, local? }` and returns numbered results, each with `source`, `method`, `score`, `metadata`, and `content`, plus `unsearched`, an optional `window`, and diagnostics. Selection (`datasourceIds`, `methods`, `local`) narrows the eligible method list **before** any backend is invoked. A query requires a completed refresh (`index-not-ready` otherwise), and `strict: true` refuses to answer from a stale index (`stale-index`). `autorag.search_datasource_<id>` searches a single connection the same way.

The agent and MCP Lite therefore read the same merged evidence; the agent additionally curates it into a plain answer whose inline citations become numbered results.
