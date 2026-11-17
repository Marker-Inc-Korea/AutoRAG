---
title: "Architecture"
description: "How AutoRAG Agent fits together: the Pi host boundary, what each src area owns, the data flow of one search, and the on-disk state layout."
---

# Architecture

This page maps AutoRAG Agent's moving parts so you can find the module that owns a behavior before you change it. It covers the runtime boundary AutoRAG borrows from Pi, what each `src/` area owns, the steps one search runs, and where state lands on disk.

AutoRAG 2.0 lives at the repository root and is the TypeScript/Bun librarian agent. The old Python AutoML tool is a separate tree under `legacy/` and is not described here; see [Development workflow](development.md#repository-layout).

## Core values

Three values drive every design decision. A change that conflicts with one of them should be rejected or reshaped.

1. **Never migrate your data to search it.** AutoRAG federates CLI-owned stores (`lazykatok`, `discrawl`, crawlers) in place. There is no forced ingestion into a central index and no third-party server holding a copy of the corpus; results carry source-native identities such as `/kakao/<instance>/chunks/<chunk>`, and secrets stay with the tool that owns them.
2. **Just works, no RAG degree required.** A non-developer installs it and it works with minimal configuration: no pipeline tuning, no vector-database operations, no OpenAI key. The local MinSync embedder (Qwen3-Embedding-0.6B) and MinSync auto-install hide the RAG plumbing.
3. **Fast by design.** One configured model owns the whole loop. Retrieval runs locally over MinSync CDC chunks (sparse / vector / hybrid), and interactive search is optimized for low latency across several model turns.

See [Design notes](design-notes.md) for why in-place federation is the durable differentiator.

## The Pi host boundary

AutoRAG is a customized [Pi](https://github.com/earendil-works/pi-mono) agent. `@earendil-works/pi-coding-agent` is a runtime dependency, and `AutoRAGAgent` (`src/agent/agent.ts`) configures the Pi agent loop into a librarian.

- Pi owns the runtime: the model provider and the credentials the user stored (`/login`, `auth.json` under the Pi agent directory), session persistence and resume, extension loading, lifecycle events, and the built-in file tools.
- AutoRAG registers only its domain tools (bash-based reading plus the retrieval, datasource, memory, and duplicate tools) and keeps orchestration outside the host: Jev routing and judgment, datasource orchestration, MinSync/Jikji preparation, retrieval memory, the two-phase search, and structured result emission.
- The distributed package assumes no private provider. The model and provider come from the user's authenticated runtime; the first launch is owned by Pi's native `/login` and `/model` flows, so the local runtime is never forced before the user can sign in.

### Terminal UI

`autorag tui` hosts Pi's interactive mode (`InteractiveMode` from `@earendil-works/pi-coding-agent`) on the AutoRAG librarian; there is no separate AutoRAG renderer. Pi owns the terminal UI and its native in-session commands — `/login`, `/logout`, `/model`, `/resume`, `/new`, `/tree`, `/compact`, and `/settings` — so provider sign-in and model selection work on first launch even before an AutoRAG model is configured. AutoRAG runs the same two-phase search in that session, so resume and two-phase search keep working together. AutoRAG suppresses Pi's `run pi update` banner and instead checks the published `@autorag/librarian` release.

## Data flow of one search

One librarian turn moves through these stages. Each is a place to look when behavior is wrong.

1. **Routing (Jev).** `src/agent/query-routing.ts` asks the decision model which branch handles the question: `local`, `organize` (file and folder actions), `extra` (a knowledge answer or any other action), or `config` (the agent's own settings). Leaving local search requires a branch probability at or above `NON_LOCAL_ROUTE_PROBABILITY_THRESHOLD` (0.75); otherwise the run falls back to `local`.
2. **Decomposition.** A question that needs several searches is split into at most five separate queries (`src/agent/query-decomposition.ts`) when Jev's `noul` probability reaches `DECOMPOSE_PROBABILITY_THRESHOLD` (0.5).
3. **Datasource selection.** Jev answers one `noul` per configured datasource, plus the reserved public-web candidate, about whether that source should be searched before the fast answer (`src/agent/query-routing.ts`).
4. **Retrieval.** `RetrievalMethodRegistry` (`src/retrieval/registry.ts`) holds the registered methods; `ParallelRetriever` fans out across them and `ResultMerger` (`src/retrieval/merger.ts`) merges, normalizes, and deduplicates by source. Datasource results pass through `filterDatasourceScope` before the merge. Every returned source is registered in the run's `EvidenceLedger` (`src/agent/evidence-ledger.ts`) and printed beside the result as an `[eN]` id.
5. **Reading and judgment.** The librarian opens promising sources directly with real paths (the `bash` tool), assesses sufficiency and freshness, and resolves conflicts.
6. **The fast answer, then verification.** The model writes a fast first answer; Jev's follow-up check (`FOLLOW_UP_PROBABILITY_THRESHOLD`, 0.5) decides whether a verification phase corrects or completes it. The final answer is the model's last assistant message when it ends its turn without a tool call.
7. **Curation.** `deriveResultsFromAnswer` (`src/agent/answer-citations.ts`) resolves the model's inline `[eN]` / `[file:<absolute path>]` citations against the ledger, rewrites the markers to `[1]`, `[2]`, … in order of first appearance, and derives one result per cited source.
8. **Memory.** After the final answer, cited evidence is mapped back to the method that surfaced it and judged in one batched Jev call (`src/agent/evidence-judgment.ts`). Evidence at or above `EVIDENCE_SUPPORT_THRESHOLD` (0.7) is stored in the retrieval memory; everything below is discarded. Memory is advisory reference and never reorders results.

Search, refresh, and MCP retrieve never index: a query only reads prebuilt indexes. Indexing runs in `refresh`/`watch` (see [Indexing overview](../indexing/overview.md)).

## Source layout

Every area under `src/` owns one concern. The agent loop orchestrates them; they do not depend on the CLI.

| Area | Owns |
|---|---|
| `src/agent/` | `AutoRAGAgent` and the agent loop: the tool surface, system prompt, two-phase search, query routing and decomposition, the evidence ledger, answer citations, and search-progress tracking |
| `src/retrieval/` | Retrieval types, the method registry, `ParallelRetriever`/`ResultMerger`, the reranker seam, time windows, scope filtering, chunk location |
| `src/minsync/` | The MinSync retrieval method, binary resolution and the version floor, install, the workspace/staging sync, and embedder configuration |
| `src/memory/` | Retrieval memory: judged evidence, curated results, similar past questions, long-term insights, and the `check_memory` tool |
| `src/datasource/` | Datasource skill contracts, scope filtering, diagnostics, the connector/chunk-store framework, crawler and connector skills, and the built-in skills under `src/datasource/skills/` |
| `src/parser/` | Document parsing (kordoc by default for HWP/PDF/DOCX/XLSX), OCR engine selection, spreadsheets, and the page index |
| `src/mirror/` | Parsed-mirror sync, page sidecars, path mapping, atomic writes, and cloud fingerprinting |
| `src/find/` | Query planning and ranking for name/content discovery |
| `src/cli/` | The `autorag` CLI: argument parsing, config resolution, every `commands/*` implementation, and the TUI entry |
| `src/mcp/` | The `autorag-mcp` MCP server and its HTTP transport, preflight, and reloading-lite surface |
| `src/cloud/` | The hosted model provider, OAuth login and token storage, model listing, and the Jev backend |
| `src/jikji/` | The optional Jikji find-first discovery client, installer, and answer-pack parsing |
| `src/fsearch/` | The macOS/Linux `fsearch-cli` client: per-workspace DB, watch daemon, and the slow-walk fallback |
| `src/everything/` | Windows-only bundled Everything extraction/verification and the per-workspace instance client |
| `src/calculator/` | Exact decimal arithmetic plus calendar and clock math for the `calculator` tool |
| `src/dupey/` | Duplicate and near-duplicate document family detection |
| `src/convert/` | Offline file conversion through the pinned `filee-cli` |
| `src/profiles/` | Plain Pi sessions for automation profiles: profile agent factory, session model resolution, and the loop guard |
| `src/automation/` | File-triggered automations: the watch daemon, rule dispatch, and agent-rule execution |
| `src/web/` | Internet `web_search` (provider chain and query parsing) and `web_fetch` (URL render pipeline) |
| `src/embedding/` | Embedding backend resolution: managed, HTTP, in-process, and daemon |
| `src/filesystem/` | Source-path normalization, path exclusions, name search, file locks, and cloud placeholders |
| `src/process/` | Portable process spawning, lifecycle, tree termination, and abort reasons |
| `src/observability/` | Run logging |
| `src/config/`, `src/language.ts`, `src/core.ts`, `src/index.ts`, `src/profiles.ts` | Home resolution, curated language tags, and the public package entry points |

## On-disk state

Two roots hold mutable state. Neither ever touches source documents; indexing writes only under them.

### AutoRAG home

`$AUTORAG_HOME` defaults to `~/.autorag`; set `AUTORAG_HOME` to move it. It holds machine-wide state, not corpus state:

| Path | Holds |
|---|---|
| `config.json` | The agent config (datasources, model, search roots). Override the file with `AUTORAG_CONFIG` or `--config` |
| `memory.json` | Retrieval memory: judged evidence, similar past questions, and long-term insights |
| `automations.json` | File-triggered automation rules |
| `attachments/discord/<connection>/…` | Discord attachments the agent saved |
| `tools/filee-cli/<version>/` | The pinned `filee-cli` binary |

### Workspace

`<workspace>/.autorag/` holds corpus-local state for the configured search roots. The default workspace is the current directory; set `--workspace` or `AUTORAG_WORKSPACE` to move it.

| Path | Holds |
|---|---|
| `parsed/files/<sha256>.md` | Parsed markdown mirror of one source file |
| `parsed/pages/<hash>.json` | Page-index sidecar beside a mirror entry |
| `parsed/stage/` | MinSync staging area used during a sync |
| `minsync/` | The MinSync workspace: `files/` (managed staging copy) and `.minsync/{config.toml,cursor.json,autorag-embedding-identity.json}` |
| `bin/minsync`, `bin/jikji` | Cached external binaries |
| `refresh-complete.json`, `refresh-progress.json` | Refresh readiness and progress markers |
| `fsearch/fsearch.db`, `fsearch/watch.pid` | The fsearch-cli database and its watch daemon pid file |
| `everything/<version>/<arch>/` | The extracted Windows Everything instance |
| `datasources/<skill>/<instance>/chunks.json` | AutoRAG-managed chunk stores for connector skills that own storage |

A `minSync.workspacePath` override moves the MinSync state elsewhere: with the default, `cursor.json` lives at `<workspace>/.autorag/minsync/.minsync/cursor.json`, while the live-E2E runner sets `minSync.workspacePath` to the workspace root, putting it at `<workspace>/.minsync/cursor.json`.

## Related pages

- [Installation](../guide/installation.md) and [Configuration file](../reference/configuration.md) for the operator surface.
- [Retrieval methods](../retrieval/overview.md), [Indexing overview](../indexing/overview.md), and [Evidence and citations](../retrieval/evidence-and-citations.md) for the subsystems in the flow above.
- [Terminal UI and search](../agents/librarian.md) for the interactive session.
- [Development workflow](development.md) and [Adding a datasource](new-datasource.md) to start changing the code.
