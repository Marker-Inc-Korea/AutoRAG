# AutoRAG — Pi-Powered Librarian Agent

## Purpose

AutoRAG is an **over-powered librarian agent** for **document collections** — PDFs, wikis, notes, research papers, knowledge bases, and any unstructured text corpus. It is a customized [Pi](https://github.com/earendil-works/pi-mono) agent: the Pi agent loop configured into a read-only librarian, used through one library/programmatic API.

**Primary target**: non-code document retrieval (manuals, legal docs, internal wikis, meeting notes, research literature).
Code repositories are a secondary use case — Pi's built-in grep/find already handle code search well.
AutoRAG's value is in the retrieval methods and curation layer that sit *on top* of raw search.

## Why AutoRAG Exists

Raw search tools return file paths and matching lines. A human still has to open each file, read the context, decide what's relevant, and synthesize an answer. AutoRAG eliminates that entire workflow:

1. **Search** across multiple retrieval methods (grep, vector, BM25, hybrid — pluggable)
2. **Read** the promising files itself
3. **Curate** — extract key insights, not raw lines
4. **Deliver** numbered knowledge units with no file paths exposed
5. **Learn** — remember which methods worked and adapt strategy over time

## Built-in Tools (Pi)

AutoRAG reuses Pi's built-in tools as its foundation:

| Tool | What it does | When to use |
|------|-------------|-------------|
| `grep` | Search **file contents** for a pattern (regex or literal). Returns matching lines with file paths and line numbers. Uses ripgrep. | Find specific text, function names, error messages, config values *inside* files |
| `find` | Find **files by name/path** using glob patterns. Returns file paths only — no content. Uses fd. | Discover files by extension (`*.pdf`), name pattern, or directory structure |
| `read` | Read **file contents** with optional line range | Examine a specific file after grep/find identified it |
| `ls` | List **directory contents** | Explore folder structure before narrowing a search |
| `check_memory` | Query **past search outcomes** | See which methods/queries succeeded before, adapt strategy |

**grep vs find in one sentence**: `grep` searches *inside* files for content; `find` searches *for* files by name.

## Architecture

```
Agent Tools                AutoRAGAgent (customized Pi agent)
┌──────────────────┐      ┌──────────────────────────────────┐
│ grep (content)   │      │ Memory System (query history)     │
│ find (files)     │ ───▶ │ Curation Layer (LLM extraction)   │
│ read (file read) │      │ check_memory (adaptive strategy)  │
│ ls (directory)   │      │ Manifest System (indexed stores)  │
└──────────────────┘      │ Feedback Loop (learn from usage)  │
                          │ Retrieval Registry (pluggable)    │
                          │ Result Merger (cross-method)      │
                          └──────────────────────────────────┘
```

## Retrieval Methods

AutoRAG is designed for **multi-method retrieval** — different methods for different document types:

| Method | Status | Best for |
|--------|--------|----------|
| posix (real directories) | Active | Plain text, docs, config files — content search over configured source directories |
| find (Pi built-in real directories) | Active | File discovery by name/glob over configured source directories |
| MinSync vector (semantic) | Active | Incrementally indexed semantic retrieval over parsed document mirrors |
| Vector (semantic) | Planned | Other dense-document backends, conceptual queries, "find similar to X" |
| BM25 (keyword) | Planned | Keyword-heavy search, term frequency ranking |
| Hybrid (vector+BM25) | Planned | Best-of-both fusion with score normalization |

The `RetrievalMethodRegistry` and `ResultMerger` are live: the real-directory `posix` method (`src/retrieval/methods/posix.ts`) is registered and routed through `ParallelRetriever` + `ResultMerger`. New methods (vector/BM25/hybrid) implement the `RetrievalMethod` interface and plug into the same pipeline.

MinSync is one vector retrieval method in that pipeline, especially useful for incremental indexing and semantic search. It should not replace real-directory `ls`/`grep`/`find` or become "the" default search surface. The AutoRAG orchestrator agent should consult memory and the query shape, then use every appropriate tool path together: Pi built-in navigation/content search for exact text, filenames, and layout-aware exploration; MinSync for indexed semantic evidence; and future BM25/hybrid methods when those are available. Each path is a tool/retrieval method feeding curation, not a privileged backend that hides the others.

Jikji is intentionally not a retrieval method in AutoRAG. It is an optional file-map and indexing preparation layer (`jikji prepare`) that can inform exploration, while query answering still flows through AutoRAG/Pi search and read tools plus registered AutoRAG retrieval methods.

## Directory Access

AutoRAG navigates document collections through normal source directories scoped to the configured `searchPaths`. The Pi agent loop uses the caller-provided search/read tools (e.g. `grep`, `find`, `read`, `ls`) plus `check_memory`, and finalizes through the `emit_autorag_results` structured tool. Retrieval and parsed mirror indexing use opaque root-relative source identifiers such as `/docs/report.md` for curation and feedback mapping; no virtual workspace layer is created.

- **Tool surface** — the agent runs with the caller-provided tools plus `check_memory` and `emit_autorag_results`. Mutating editors (`edit`/`write`) are excluded because AutoRAG is read-only.
- **Real-directory posix method** — `src/retrieval/methods/posix.ts` recursively scans configured `searchPaths`, scores files by match count and depth, and returns opaque root-relative source identifiers.
- **Parsed mirrors** — `AutoRAGAgent.refresh()` parses supported files directly from configured source directories into `.autorag/parsed`; MinSync indexes those parsed mirrors unchanged.
- **Jikji preparation** — `AutoRAGAgent.prepareJikji()` runs `jikji prepare` over configured source directories only. AutoRAG does not call `jikji find` or merge Jikji answers as retrieval results.

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

**Caller sees:**
```
[1] Revenue Summary — Q3 revenue grew 23% YoY to $4.2M, driven by enterprise contracts. (pages 3-5)
[2] Risk Factors — Three new risk factors added: supply chain, regulatory, talent retention. (pages 12-14)
```

**Caller does NOT see:** file paths, retrieval method names, raw grep output.

## Memory System (Self-Evolving)

AutoRAG remembers past search outcomes across sessions:
- Tracks which queries + methods succeeded or failed
- Prioritizes methods that historically work for similar queries
- `check_memory` tool lets the LLM query this history before searching
- Feedback loop: callers mark results as useful/not-useful → improves future searches

Over time, AutoRAG learns which retrieval methods work best for which types of queries in your specific document collection. A fresh AutoRAG tries everything; a seasoned one goes straight to what works.

## Feedback Flow

1. Caller references results by session ID + number (e.g., session "abc", [1,3] useful)
2. Agent resolves numbers → session registry (populated from `emit_autorag_results` details) → source paths
3. Source paths → memory entries updated (useful/not_useful)
4. Memory informs future search strategy

## Files

| File | Role |
|------|------|
| `src/agent/agent.ts` | AutoRAGAgent class — the customized Pi agent and library API |
| `src/agent/emit-results-tool.ts` | `emit_autorag_results` terminating tool that returns curated results as typed details |
| `src/agent/system-prompt.ts` | System prompt builder for the librarian agent |
| `src/memory/memory.ts` | Feedback persistence and method priority scoring |
| `src/memory/renderer.ts` | Memory context renderer for system prompt |
| `src/memory/check-memory-tool.ts` | check_memory tool (pi-agent-core AgentTool) |
| `src/manifest/loader.ts` | YAML/JSON manifest loader for indexed data stores |
| `src/retrieval/types.ts` | Core retrieval type definitions |
| `src/retrieval/registry.ts` | Method registry for multi-method orchestration |
| `src/retrieval/merger.ts` | Cross-method result merging and deduplication |
| `src/retrieval/methods/posix.ts` | Real-directory `posix` RetrievalMethod |
