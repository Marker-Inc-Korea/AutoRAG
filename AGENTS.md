# AutoRAG — Pi-Powered Librarian Agent

## Purpose

AutoRAG is an **over-powered librarian agent** for **document collections** — PDFs, wikis, notes, research papers, knowledge bases, and any unstructured text corpus. It is a customized [Pi](https://github.com/earendil-works/pi-mono) agent: the Pi agent loop configured into a librarian, used through one library/programmatic API (and a thin CLI).

Searches use a mandatory two-tier workflow: a parent orchestrator delegates document exploration to explorer agents. The roles and providers are independently configurable from the user's authenticated runtime; the distributed package does not assume a private provider.

**Primary target**: non-code document retrieval (manuals, legal docs, internal wikis, meeting notes, research literature).
Code repositories work too. AutoRAG's value is in the exploration + retrieval methods + curation layer that sit *on top* of raw search.

## Why AutoRAG Exists

Raw search tools return file paths and matching lines. A human still has to open each file, read the context, decide what's relevant, and synthesize an answer. AutoRAG eliminates that entire workflow:

1. **Delegate exploration** to gpt-5.6-luna explorers through `pi-subagents`
2. **Search** across multiple retrieval methods (BM25, vector/MinSync, datasource skills — pluggable)
3. **Read** the promising files through explorer tasks
4. **Judge and curate** — extract key insights, resolve conflicts, and assess freshness
5. **Deliver** numbered knowledge units grounded in the sources
6. **Learn** — remember which methods worked and adapt strategy over time

## Agent Tools

The parent orchestrator uses `pi-subagents` to assign explorer tasks. Explorer tasks are read-only and use `read`, `grep`, `find`, and `ls`; process-bound retrieval and finalization stay with the parent orchestrator:

| Tool | What it does | When to use |
|------|-------------|-------------|
| `read`, `grep`, `find`, `ls` | Read-only filesystem discovery and document reading with real paths | Explorer tasks only |
| `jikji_find` | Runs `jikji find ROOT "query"` and enforces the returned answer-pack policy (`handoff_action`, `tool_call_policy`, `agent_should_not_rerank`) | Parent orchestrator's first local-discovery seed when Jikji is configured |
| `search_all_documents` | Fan-out across configured retrieval methods and merge/rank a bounded candidate pack | Parent orchestrator seed retrieval |
| `search_bm25_documents` | Lexical BM25 ranking over parsed document mirrors | Parent orchestrator seed retrieval |
| `search_minsync_documents` | MinSync semantic/vector retrieval over parsed mirrors | Parent orchestrator seed retrieval |
| `search_datasource_documents` | Search authorized external datasource skills | Parent orchestrator seed retrieval; server-bound access |
| `check_memory` | Query past search outcomes | Parent orchestrator strategy |
| `load_datasource_skill` | Load instructions for an authorized datasource skill | Parent orchestrator only |
| `emit_autorag_results` | Terminating tool that returns curated results | Parent orchestrator's final action |

The parent may create a bounded retrieval seed pack, but must delegate the underlying document reading to one or more explorers. Explorers return evidence handoffs; they do not make the final judgment or call the terminating tool.

## Architecture

```
Agent Tools                 AutoRAGAgent (customized Pi agent)
┌──────────────────┐       ┌──────────────────────────────────┐
│ read/grep/find/ls │       │ Memory System (query history)     │
│ seed retrieval    │  ───▶ │ Curation Layer (LLM extraction)   │
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

The `RetrievalMethodRegistry` and `ResultMerger` are live: configured methods are registered and routed through `ParallelRetriever` + `ResultMerger`. New methods implement the `RetrievalMethod` interface and plug into the same pipeline. Plain-directory content search is no longer a registered retrieval method — explorers do that directly with `read`/`grep`/`find`/`ls`.

Jikji is intentionally not a retrieval method. It is an optional **find-first local-discovery** layer: when configured, AutoRAG calls `jikji find ROOT "query" --json` via a policy-aware `jikji_find` tool as the first local-discovery action. The tool parses and validates the upstream answer-pack and honors its `handoff_action` (`direct_use` / `jikji_retry` / `raw_fallback_after_retry`), `tool_call_policy` (`stop_after_find`, `forbidden_tools`, `allowed_followups`), and `agent_should_not_rerank`. Explorer `read`/`grep`/`find`/`ls` discovery is the fallback only when the answer-pack permits raw fallback (`raw_fallback_after_retry`, after the required retry) or when Jikji is unavailable/unconfigured. `prepare`/`refresh` remain for indexing only and do not answer queries directly.

Datasource skills are retrieval-method factories plus indexing hooks for external, server-configured data sources. They remain inside the same pipeline — `RetrievalMethodRegistry` → `ParallelRetriever` → `DatasourceResultFilter` → `ResultMerger`. Datasource access is default-deny and server-bound: LLM tool arguments cannot grant `allowedTags` or `allowedScopes`, and `search_datasource_documents` exposes only `{ query, topK?, scope? }` where `scope` can only narrow trusted access.

The first concrete datasource is KakaoTalk through the external `katok` CLI. AutoRAG never reads KakaoTalk databases directly; failures surface as diagnostics, and remote embedding egress settings are rejected before the CLI is spawned.

## Directory Access

gpt-5.6-luna explorers navigate document collections through read-only `read`/`grep`/`find`/`ls`; each explorer is assigned exactly one normalized configured search root as its `cwd`. The top-level `subagent` invocation sets `agentScope: "user"` and `artifacts: false` exactly once for single, `tasks`, `chain`, or `parallel` dispatch; nested explorer task items omit both fields. Project-local `.pi-subagents` debug artifacts are therefore disabled. The gpt-5.6-sol orchestrator owns `check_memory`, Jikji, datasource, and `search_*` seed retrieval, delegates those seeds through `pi-subagents`, judges the returned evidence, and finalizes through the `emit_autorag_results` structured tool. Real file paths are visible to explorers and may appear in curated results and their source mapping; the orchestrator must not bypass the required delegation with direct document reading.

Each explorer `task` contains an Assignment V1 block — a sentinel-wrapped JSON body (`<<<AUTORAG_ASSIGNMENT_V1>>>` … `<<<END_AUTORAG_ASSIGNMENT_V1>>>`) with exactly `originalQuery`, `method`, and `queryVariants`, followed by canonical role lines requiring `retrievedAt` and temporal metadata. A legacy labeled format (`Original query:`, `Selected retrieval method:`, `Query variants:`) is accepted for compatibility. Missing or null top-level `artifacts`, `agentScope`, and leaf `model` fields are safely autofilled before validation; explicit wrong values (`artifacts: true`, `agentScope: "project"`, a non-configured model) remain rejected. Diagnostics (`list`, `get`, `models`, `status`, `doctor`) are separate from launch dispatch and never receive these defaults. There is no single-agent fallback. Rejected dispatches return a stable coded error with an `exactFix` string; see [docs/subagent-orchestration.md](docs/subagent-orchestration.md) for the full table.

Durable Pi models, settings, and sessions stay under `~/.autorag/pi-agent`; corpus indexes remain workspace-local under `<workspace>/.autorag`.

Subagent role separation, dispatch inputs, explorer handoff metadata, and the mandatory `pi-subagents` workflow are defined in [docs/subagent-orchestration.md](docs/subagent-orchestration.md). The `gpt-5.6-sol` orchestrator owns judgment, sufficiency, conflicts, freshness, timing, follow-up assignments, and final curation; `gpt-5.6-luna` explorers provide broad retrieval/read evidence, including weak candidates and temporal metadata. A single-agent fallback is forbidden.

- **Tool surface** — explorer tasks run with read-only `read`, `grep`, `find`, and `ls`; the parent orchestrator owns `check_memory`, `jikji_find` (when configured), the `search_*` retrieval seed tools, `load_datasource_skill`, and `emit_autorag_results`, then delegates source reading through `pi-subagents`.
- **Parsed mirrors** — `AutoRAGAgent.refresh()` parses supported files from configured source directories into `.autorag/parsed`; BM25 and MinSync index those parsed mirrors.
- **Jikji find-first** — when Jikji is configured, `jikji_find` runs `jikji find ROOT "query" --json` as the first local-discovery action and enforces the returned `handoff_action` / `tool_call_policy` / `agent_should_not_rerank`; explorer `read`/`grep`/`find`/`ls` discovery is the fallback only when the pack permits raw fallback or Jikji is unavailable. `prepare`/`refresh` remain for indexing only; AutoRAG-managed prepare runs with `--no-agent-rules` by default so it never rewrites the consumer repo's `AGENTS.md`/`CLAUDE.md`/`.cursorrules`. An explicit `writeAgentRules: true` opt-in re-enables upstream routing-block injection.
- **External tool auto-install** — MinSync and Jikji binaries are cached under `<workspace>/.autorag/bin`. MinSync auto-installs from verified GitHub release assets by default (`minSync.autoInstall: false` opts out). Jikji auto-installs the `jikji-cli` crate from crates.io via cargo by default (`jikji.autoInstall: false` opts out; requires the Rust toolchain). New `autorag init` configs enable Jikji by default (`jikji: {}`). The KakaoTalk `katok` CLI remains a manual, optional install. All three degrade gracefully when missing.
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
| `src/agent/bash-tool.ts` | Parent-side gated POSIX bridge; explorer tasks use read-only `read`/`grep`/`find`/`ls` |
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
| `src/agent/search-datasource-tool.ts` | `search_datasource_documents` tool with model-safe `{ query, topK?, scope? }` parameters |
