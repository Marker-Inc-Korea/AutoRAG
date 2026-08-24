# AutoRAG — Pi-Powered Librarian Agent

## Developer Commands

The repository root includes a `Makefile` for AutoRAG 2.0 validation:

- `make test` / `make test-all` — run the complete test suite.
- `make test-macos` — run the complete suite and require a macOS host.
- `make test-windows` — run the complete suite on a Windows host.
- `make test-linux` — run lint, typecheck, the complete suite, and build in a Docker container.
- `make lint`, `make typecheck`, `make build` — run individual checks.
- `make ci` — run the normal local lint, typecheck, complete test, and build sequence.

Docker can reproduce the Linux job on macOS, Linux, or Windows hosts. The
`test-linux` target uses an isolated container volume for `node_modules`, so it
does not replace host-native dependencies, and pins `linux/amd64` to match
GitHub's Ubuntu runner and the available Tantivy native binding. Windows
containers require a Windows kernel, so Windows compatibility is run natively
from Git Bash/MSYS2 with `make test-windows` (or directly with
`bun run test:windows`) and verified by the `windows-latest` GitHub-hosted
runner.

## Purpose

AutoRAG is an **over-powered librarian agent** for **document collections** — PDFs, wikis, notes, research papers, knowledge bases, and any unstructured text corpus. It is a customized [Pi](https://github.com/earendil-works/pi-mono) agent: the Pi agent loop configured into a librarian, used through one library/programmatic API (and a thin CLI).

Searches run in one agent loop: the librarian chooses retrieval methods, reads source files directly, judges the evidence, and curates structured results. The model and provider come from the user's authenticated runtime; the distributed package does not assume a private provider.

AutoRAG itself is the specialized librarian agent. It uses one configured
model for the whole search loop, so model selection should favor reliable tool
calling and structured output. High TPS and low first-token latency improve
interactive search speed because retrieval commonly spans several model turns;
they do not make the underlying BM25, MinSync, Jikji, filesystem, or indexing
operations faster.

**Primary target**: non-code document retrieval (manuals, legal docs, internal wikis, meeting notes, research literature).
Code repositories work too. AutoRAG's value is in the exploration + retrieval methods + curation layer that sit *on top* of raw search.

## Why AutoRAG Exists

Raw search tools return file paths and matching lines. A human still has to open each file, read the context, decide what's relevant, and synthesize an answer. AutoRAG eliminates that entire workflow:

1. **Search** across multiple retrieval methods (BM25, vector/MinSync, datasource skills — pluggable)
2. **Read** promising source files directly with the built-in bash tool
3. **Judge and curate** — extract key insights, resolve conflicts, and assess freshness
4. **Deliver** numbered knowledge units grounded in the sources
5. **Learn** — remember which methods worked and adapt strategy over time

## Agent Tools

The librarian agent owns the full workflow:

| Tool | What it does | When to use |
|------|-------------|-------------|
| `bash` | Filesystem discovery and document reading with real paths (`ls`, `find`, `grep`, `cat`, etc.) | Direct source verification |
| `jikji_find` | Runs `jikji find ROOT "query"` and returns a policy-aware answer pack | Optional local discovery |
| `search_all_documents` | Fan-out across configured retrieval methods and merge/rank candidates | Combined retrieval |
| `search_bm25_documents` | Lexical BM25 ranking over parsed document mirrors | Exact-term retrieval |
| `search_minsync_documents` | MinSync semantic/vector retrieval over parsed mirrors | Semantic retrieval |
| `search_datasource_documents` | Search authorized external datasource skills | Server-bound datasource retrieval |
| `check_memory` | Query past search outcomes | Adaptive strategy |
| `load_datasource_skill` | Load instructions for an authorized datasource skill | Datasource-specific searches |
| `emit_autorag_results` | Terminating tool that returns curated results | Final action |

## Architecture

```
Agent Tools                 AutoRAGAgent (customized Pi agent)
┌──────────────────┐       ┌──────────────────────────────────┐
│ bash read/search  │       │ Memory System (query history)     │
│ retrieval tools   │  ───▶ │ Curation Layer (LLM extraction)   │
│ search_bm25      │       │ check_memory (adaptive strategy)  │
│ search_minsync   │       │ Manifest System (indexed stores)  │
│ search_datasource│       │ Retrieval Registry (pluggable)    │
│ check_memory     │       │ Result Merger (cross-method)      │
└──────────────────┘       │ Feedback Loop (learn from usage)  │
                           └──────────────────────────────────┘
```

## Retrieval Methods

AutoRAG is designed for **multi-method retrieval** — different methods for different document types:

| Method | Status | Best for |
|--------|--------|----------|
| BM25 (keyword) | Active | Keyword-heavy search, term frequency ranking over parsed mirrors |
| MinSync vector (semantic) | Active | Incrementally indexed semantic retrieval over parsed document mirrors |
| Datasource skills | Active | External server-configured sources (e.g. KakaoTalk via `katok`) |
| Vector (other backends) | Planned | Other dense-document backends, "find similar to X" |
| Hybrid (vector+BM25) | Planned | Best-of-both fusion with score normalization |

The `RetrievalMethodRegistry` and `ResultMerger` are live: configured methods are registered and routed through `ParallelRetriever` + `ResultMerger`. New methods implement the `RetrievalMethod` interface and plug into the same pipeline. Plain-directory content search is handled directly through the agent's `bash` tool.

Jikji is intentionally not a retrieval method. It is an optional local-discovery layer: AutoRAG calls `jikji find ROOT "query" --json` through `jikji_find`, parses the upstream answer pack, and exposes `handoff_action`, `tool_call_policy`, and `agent_should_not_rerank` to the librarian. Direct `bash` reading remains available so Jikji never prevents source verification. `prepare`/`refresh` remain for indexing only and do not answer queries directly.

Datasource skills are retrieval-method factories plus indexing hooks for external, server-configured data sources. They remain inside the same pipeline — `RetrievalMethodRegistry` → `ParallelRetriever` → `DatasourceResultFilter` → `ResultMerger`. Datasource access is default-deny and server-bound: LLM tool arguments cannot grant `allowedTags` or `allowedScopes`, and `search_datasource_documents` exposes only `{ query, topK?, scope? }` where `scope` can only narrow trusted access. Results are not redacted — traceability is preferred over opacity, so pair AutoRAG with a local LLM when privacy matters.

CLI-backed datasources own their archive, lexical index, and vectors: KakaoTalk through the external `katok` CLI, and **Discord** through the external [`discrawl`](https://github.com/openclaw/discrawl) CLI. AutoRAG only spawns them and maps results. AutoRAG never reads KakaoTalk databases directly; failures surface as diagnostics, and remote embedding egress settings are rejected before the CLI is spawned.

External crawler-backed skills cover **WhatsApp** (wacrawl), **Telegram** (telecrawl), **Slack** (slacrawl), and **Notion** (notcrawl); each crawler owns its archive, sync, credentials, and FTS search while AutoRAG provides bounded process execution, diagnostics, and retrieval mapping. The remaining connector-backed datasource skills use the shared framework (`src/datasource/connector.ts`, `chunk-store.ts`, `connector-skill.ts`): **GitHub**, **Google Drive**, **Gmail**, **local mail export**, **Obsidian** (vault via external `qmd` CLI: incremental + BM25 + semantic), **RSS/news**, and **Spotlight**. Results remain traceable and datasource access stays default-deny. Manual QA harnesses live in `scripts/manual-qa/` (see `docs/manual-qa-datasources.md`).

## Directory Access

The AutoRAG librarian navigates document collections directly with `bash`, using real paths for discovery and reading. Retrieval tools return bounded candidates; the librarian opens the source material, assesses sufficiency and freshness, resolves conflicts, and finalizes with `emit_autorag_results`.

Model authentication stays with the configured provider or authenticated local runtime; corpus indexes remain workspace-local under `<workspace>/.autorag`.

- **Tool surface** — the librarian owns `bash`, `check_memory`, `jikji_find`, the `search_*` retrieval tools, `load_datasource_skill`, and `emit_autorag_results`.
- **Parsed mirrors** — `AutoRAGAgent.refresh()` parses supported files from configured source directories into `.autorag/parsed`; BM25 and MinSync index those parsed mirrors.
- **Jikji discovery** — `jikji_find` runs `jikji find ROOT "query" --json` and returns the answer pack to the librarian; direct file reading remains available. `prepare`/`refresh` remain for indexing only; AutoRAG-managed prepare runs with `--no-agent-rules` by default so it never rewrites the consumer repo's `AGENTS.md`/`CLAUDE.md`/`.cursorrules`. An explicit `writeAgentRules: true` opt-in re-enables upstream routing-block injection.
- **External tool auto-install** — MinSync and Jikji binaries are cached under `<workspace>/.autorag/bin`. MinSync auto-installs from verified GitHub release assets by default (`minSync.autoInstall: false` opts out). Jikji auto-installs the `jikji-cli` crate from crates.io via cargo by default (`jikji.autoInstall: false` opts out; requires the Rust toolchain). New `autorag init` configs enable Jikji by default (`jikji: {}`). The KakaoTalk `katok` and Discord `discrawl` CLIs remain manual, optional installs (`brew install openclaw/tap/discrawl`). All three degrade gracefully when missing.
- **Datasource skills** — `AutoRAGAgent` can register `datasourceSkills`; their retrieval methods are merged with the normal retrieval pipeline, filtered before merging by trusted datasource access, and indexed during `refresh()`.

## Usage

```typescript
import { AutoRAGAgent } from "@autorag/librarian";

const agent = new AutoRAGAgent({
  searchPaths: ["/path/to/documents"],
});
const response = await agent.searchDocuments("summarize the Q3 financial report");
console.log(response.answer);
agent.recordFeedbackByNumbers(response.sessionId, [1, 3], [2]);
```

`searchDocuments()` drives the Pi agent loop and returns a typed `SearchDocumentsResponse`; the caller consumes the structured payload directly, without parsing assistant text.

## Output Contract

**Caller sees curated, numbered knowledge units:**
```
[1] Revenue Summary — Q3 revenue grew 23% YoY to $4.2M, driven by enterprise contracts. (pages 3-5)
[2] Risk Factors — Three new risk factors added: supply chain, regulatory, talent retention. (pages 12-14)
```

Each result maps to an internal entry carrying its `source` (a real file path or datasource id), `method`, and evidence for feedback tracking. The curated `answer`/`results` are grounded in the sources; source paths may appear where relevant.

## Memory System (Self-Evolving)

AutoRAG remembers past search outcomes across sessions:
- Tracks which queries + methods succeeded or failed
- Prioritizes methods that historically work for similar queries
- `check_memory` tool lets the LLM query this history before searching
- Feedback loop: callers mark results as useful/not-useful → improves future searches

## Feedback Flow

1. Caller references results by session ID + number (e.g., session "abc", [1,3] useful)
2. Agent resolves numbers → session registry (populated from `emit_autorag_results` details) → sources
3. Sources → memory entries updated (useful/not_useful)
4. Memory informs future search strategy

## Files

| File | Role |
|------|------|
| `src/agent/agent.ts` | AutoRAGAgent class — the customized Pi agent and library API |
| `src/agent/bash-tool.ts` | Direct filesystem discovery and document-reading tool |
| `src/agent/emit-results-tool.ts` | `emit_autorag_results` terminating tool that returns curated results as typed details |
| `src/agent/system-prompt.ts` | System prompt builder for the librarian agent |
| `src/memory/memory.ts` | Feedback persistence and method priority scoring |
| `src/memory/renderer.ts` | Memory context renderer for system prompt |
| `src/memory/check-memory-tool.ts` | check_memory tool (pi-agent-core AgentTool) |
| `src/manifest/loader.ts` | YAML/JSON manifest loader for indexed data stores |
| `src/retrieval/types.ts` | Core retrieval type definitions |
| `src/retrieval/registry.ts` | Method registry for multi-method orchestration |
| `src/retrieval/merger.ts` | Cross-method result merging and deduplication |
| `src/retrieval/methods/bm25.ts` | BM25 lexical RetrievalMethod over parsed mirrors |
| `src/datasource/` | Datasource skill contracts, trusted access context, result filtering, polling metadata, diagnostics, and KakaoTalk/katok skill implementation |
| `src/datasource/connector.ts` | Connector contract + opaque-text/id sanitizers for connector-backed skills |
| `src/datasource/chunk-store.ts` | Persistent chunk store with BM25-style lexical search per skill instance |
| `src/datasource/connector-skill.ts` | Shared DatasourceSkill base composing a connector with the chunk store |
| `src/datasource/skills/` | Built-in skills: katok, discrawl (Discord), slack, notion, github, gdrive, gmail, mail-export, obsidian, rss, spotlight (+ config factory) |
| `src/agent/search-datasource-tool.ts` | `search_datasource_documents` tool with model-safe `{ query, topK?, scope? }` parameters |
