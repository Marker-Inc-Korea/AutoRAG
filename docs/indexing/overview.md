---
title: "Overview"
description: "How AutoRAG turns configured search roots into parsed mirrors and local indexes, and what state it writes under your workspace and home."
---

# Overview

This page shows how `autorag refresh` moves documents from your search roots into the local indexes an agent searches, and what each index is for. It lists the refresh methods, the workspace state directory, and the fields `autorag status` reports.

## The indexing pipeline

Indexing is model-free. No LLM is constructed, and a query never indexes.

1. **Search roots.** The configured `searchPaths` are walked. Files that no parser claims, files that exceed the parse-size cap, and duplicate copies excluded by the exact-duplicate scan are skipped.
2. **Parsed mirrors.** Every parseable source is parsed into Markdown under `<workspace>/.autorag/parsed/files/<sha256>.md`, keyed by the SHA-256 of its virtual path. Parsing is incremental and durable: unchanged, verified mirrors are never read or parsed again, and the recording of completed work is owned by MinSync's stage protocol.
3. **MinSync workspace.** Parsed mirrors are staged into `<workspace>/.autorag/minsync/files/`, and MinSync performs Change Data Capture: it chunks changed documents and encodes each chunk into dense vectors and SPLADE sparse terms over the same chunk IDs.
4. **Local indexes.** Retrieval reads these indexes: MinSync `sparse`, `vector`, and `hybrid` modes over the CDC chunks, Jikji's discovery answer packs, and a file-name index (Everything on Windows, fsearch-cli on macOS/Linux).

The MinSync datasource is the semantic core. Memory, the chat-crawler sidecars, the GitHub Gist vector index, and the endpoint external CLIs are pointed at share one embedder with MinSync. See [Embedding backends](embedding-backends.md) and [Sparse index and chunking](sparse-and-chunking.md).

Jikji is intentionally not a retrieval method: it is an optional local-discovery layer, and `prepare`/`refresh` exist for indexing only. Direct `bash` reading of source files remains available. See [Retrieval overview](../retrieval/overview.md).

## The workspace state directory

`workspacePath` is the directory that owns AutoRAG's local state; it defaults to the current directory. Indexing writes only under `<workspace>/.autorag/`, Jikji's central store, and the home directory — never into source documents.

| Path | Contents |
| --- | --- |
| `<workspace>/.autorag/parsed/files/` | Parsed Markdown mirrors, one `<sha256>.md` per source. |
| `<workspace>/.autorag/parsed/stage/` | MinSync's durable parse-stage lock, verified completions, journal, and snapshot. |
| `<workspace>/.autorag/parsed/pages/` | Per-mirror page index sidecars (`.json`), keyed to the Markdown they were measured on. |
| `<workspace>/.autorag/minsync/` | MinSync workspace: staged `files/` and MinSync's own `.minsync/` state. |
| `<workspace>/.autorag/datasources/` | Per-skill chunk stores and connector mirrors. |
| `<workspace>/.autorag/bin/` | Auto-installed MinSync and Jikji binaries. |
| `<workspace>/.autorag/everything/<version>/<arch>/` | Windows-only bundled voidtools Everything instance and its database. |
| `<workspace>/.autorag/fsearch/` | macOS/Linux fsearch-cli database and `watch.pid`. |
| `<workspace>/.autorag/models/tessdata/` | Tesseract traineddata cache, only when OCR selects Tesseract. |
| `<workspace>/.autorag/refresh-progress.json` | Live refresh progress and owner pid (display and liveness, not the engine's recovery checkpoint). |
| `<workspace>/.autorag/refresh-complete.json` | Marker that a parsed-mirror refresh has completed at least once. |
| `<workspace>/.autorag/cloud-fingerprints.json` | dupey fingerprints for cloud and network files. |

The global home (`~/.autorag`, moved by `$AUTORAG_HOME`) holds `config.json`, `memory.json`, `attachments/`, `automations.json`, and TUI sessions. `AUTORAG_CONFIG` overrides the config path.

## Refresh methods

`autorag refresh` runs every method by default. `--method` takes a comma-separated list of `minsync`, `parsed`, `datasources`, `jikji`, `everything`, `fsearch`, or `all`.

| Method | What it does |
| --- | --- |
| `parsed` | Parses changed sources into the parsed mirror and runs the exact-duplicate scan. |
| `minsync` | Runs `parsed`, then stages mirrors into the MinSync workspace and syncs the dense and SPLADE indexes. |
| `datasources` | Runs each configured datasource skill's indexing hook. |
| `jikji` | Runs `jikji prepare` over the search roots. Also runs after `minsync`. |
| `everything` | Windows only: rewrites and restarts the bundled Everything name index. |
| `fsearch` | macOS/Linux only: rebuilds the fsearch-cli name index and re-verifies its watch daemon. |
| `all` | Every method above (the default when `--method` is omitted). |

Selecting `minsync` implies `parsed`. A `parsed` or `minsync` run also refreshes the platform file-name index. See [Refresh operations](refresh-operations.md) for diagnostics and failure behavior.

To drop and rebuild: `autorag index reset --method minsync,parsed --yes` removes the parsed mirror and MinSync indexes (keeping `bin`, `datasources`, and memory), and `autorag index rebuild --method minsync` resets then re-runs a forced refresh. Without `--method`, both targets are reset.

## Status fields

`autorag status` is a model-free snapshot. It checks corpus staleness, reads the last refresh outcome, and inspects the current Jikji index for each local search root. Native Jikji errors and warnings are shown verbatim, including filesystem paths when present.

| Field | Meaning |
| --- | --- |
| `state` | The refresh outcome: `idle`, `indexing`, `success`, or `failed`. This is separate from current Jikji index health. |
| `inFlight` | A refresh is running in this process. |
| `stale` | Parsed-mirror and MinSync freshness: an index is missing, the engine is unavailable, or a source changed since the last refresh. Jikji health is reported separately. |
| `lastStartedAt` / `lastFinishedAt` | Timestamps of the last refresh, when known. |
| `counts` | `scanned`, `written`, `deleted`, `skipped` for the last run. |
| `progress.phase` | Current phase: `parsed`, `minsync`, `datasources`, `jikji`, `everything`, `fsearch`, or `finalizing`. |
| `progress.sourceFiles.total` | Total sources in the current parse pass. |
| `progress.minsync` | `synced`, `total`, and `filesSkipped` counters. |
| `progress.ownerAlive` | Whether the recorded owner pid is still alive. |
| `components` | Per-component readiness: `minsync`, `jikji`, `datasources`, `everything`, `fsearch`. |
| `jikji` | Per-root current health: `source`, `state` (`prepared`, `not-prepared`, `needs-reprepare`, `unavailable`, or `skipped`), optional `docsFailed`, and native `errors` / `warnings`. Omitted when Jikji is disabled. |
| `diagnostics` | Current freshness and Jikji health diagnostics, plus diagnostics retained from the last refresh (see [Refresh operations](refresh-operations.md)). |
| `lastError` | A path-free summary when the last refresh failed. |

Each status call checks Jikji again, so a new CLI process and an existing agent both see indexes prepared, repaired, or removed outside AutoRAG. `components.jikji` summarizes these roots as `ready`, `not-prepared`, `degraded`, `unavailable`, or `skipped`; a synchronous SDK component snapshot can still say `configured` before the first inspection. A successful refresh can therefore coexist with `needs-reprepare` or a positive `docsFailed` count. Refresh also includes these index diagnostics after running Jikji preparation.

Status does not install Jikji, create an index, repair it, or re-prepare it. It checks for an existing prepared root before invoking native `jikji doctor`; doctor can refresh internal schema metadata in an existing database. Cloud and network roots are marked `skipped`, and disabled Jikji is not inspected. A `needs-reprepare` state reports the native failure (for example, `missing required artifact row: wiki_index`) so you can decide when to run `autorag refresh --method jikji` explicitly.

## Next steps

Start from [Installation](../guide/installation.md), then read [Refresh operations](refresh-operations.md) before pointing AutoRAG at a large or partially-available collection.
