# AutoRAG — Pi-Powered Librarian Agent

## Purpose

AutoRAG is an **over-powered librarian agent** for **document collections** — PDFs, wikis, notes, research papers, knowledge bases, and any unstructured text corpus. It is a customized [Pi](https://github.com/earendil-works/pi-mono) agent: the Pi agent loop configured into a librarian, used through one library/programmatic API (and a thin CLI).

**Primary target**: non-code document retrieval (manuals, legal docs, internal wikis, meeting notes, research literature).
Code repositories work too. AutoRAG's value is in the exploration + retrieval methods + curation layer that sit *on top* of raw search.

## Why AutoRAG Exists

Raw search tools return file paths and matching lines. A human still has to open each file, read the context, decide what's relevant, and synthesize an answer. AutoRAG eliminates that entire workflow:

1. **Explore** the collection directly with a shell (bash)
2. **Search** across multiple retrieval methods (BM25, vector/MinSync, datasource skills — pluggable)
3. **Read** the promising files itself
4. **Curate** — extract key insights, not raw lines
5. **Deliver** numbered knowledge units grounded in the sources
6. **Learn** — remember which methods worked and adapt strategy over time

## Agent Tools

The librarian runs with a real shell plus the retrieval and memory tools:

| Tool | What it does | When to use |
|------|-------------|-------------|
| `bash` | Run shell commands (`ls`, `find`, `grep`, `rg`, `cat`, `head`, `sed`). Returns combined stdout/stderr, including real file paths. | Navigate the collection, search file contents, and read files directly |
| `search_all_documents` | Fan out across every configured retrieval method and merge/rank results | Fast multi-method first pass |
| `search_bm25_documents` | Lexical BM25 ranking over parsed document mirrors | Exact terms, headings, identifiers |
| `search_minsync_documents` | MinSync semantic/vector retrieval over parsed mirrors | Conceptual, meaning-based evidence |
| `search_datasource_documents` | Search authorized external datasource skills | KakaoTalk chats, etc. (server-bound access) |
| `check_memory` | Query past search outcomes | See which methods/queries succeeded before |
| `emit_autorag_results` | Terminating tool that returns curated results | Final action of every run |

AutoRAG explores the collection with `bash` and consults the retrieval methods; there is no separate builtin `grep`/`find`/`read`/`ls` tool and no `posix` retrieval method.

## Architecture

```
Agent Tools                 AutoRAGAgent (customized Pi agent)
┌──────────────────┐       ┌──────────────────────────────────┐
│ bash (shell)     │       │ Memory System (query history)     │
│ search_all       │  ───▶ │ Curation Layer (LLM extraction)   │
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

The `RetrievalMethodRegistry` and `ResultMerger` are live: configured methods are registered and routed through `ParallelRetriever` + `ResultMerger`. New methods implement the `RetrievalMethod` interface and plug into the same pipeline. Plain-directory content search is no longer a registered retrieval method — the agent does that directly with `bash`.

Jikji is intentionally not a retrieval method. It is an optional file-map and indexing preparation layer (`jikji prepare`) that is surfaced to the agent as a **navigation hint** in the system prompt; query answering flows through `bash` and the registered retrieval methods.

Datasource skills are retrieval-method factories plus indexing hooks for external, server-configured data sources. They remain inside the same pipeline — `RetrievalMethodRegistry` → `ParallelRetriever` → `DatasourceResultFilter` → `ResultMerger`. Datasource access is default-deny and server-bound: LLM tool arguments cannot grant `allowedTags` or `allowedScopes`, and `search_datasource_documents` exposes only `{ query, topK?, scope? }` where `scope` can only narrow trusted access.

The first concrete datasource is KakaoTalk through the external `katok` CLI. AutoRAG never reads KakaoTalk databases directly; failures surface as diagnostics, and remote embedding egress settings are rejected before the CLI is spawned.

## Directory Access

AutoRAG navigates document collections directly through `bash`, scoped to the configured `searchPaths` / workspace root. The Pi agent loop uses `bash` plus `check_memory` and the retrieval tools, and finalizes through the `emit_autorag_results` structured tool. Real file paths are visible to the agent and may appear in curated results and their source mapping.

- **Tool surface** — the agent runs with `bash`, `check_memory`, the `search_*` retrieval tools, `load_datasource_skill`, and `emit_autorag_results`, plus any non-reserved caller-provided tools.
- **Parsed mirrors** — `AutoRAGAgent.refresh()` parses supported files from configured source directories into `.autorag/parsed`; BM25 and MinSync index those parsed mirrors.
- **Jikji preparation** — `AutoRAGAgent.prepareJikji()` runs `jikji prepare` over configured source directories only. AutoRAG does not call `jikji find` or merge Jikji answers as retrieval results.
- **Datasource skills** — `AutoRAGAgent` can register `datasourceSkills`; their retrieval methods are merged with the normal retrieval pipeline, filtered before merging by trusted datasource access, and indexed during `refresh()`.

## Usage

```typescript
import { AutoRAGAgent } from "@autorag/librarian";
import { getModel } from "@earendil-works/pi-ai";

const agent = new AutoRAGAgent({
  model: getModel("anthropic", "claude-sonnet-4-20250514"),
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
| `src/agent/bash-tool.ts` | `bash` shell tool for agentic exploration and reading |
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
