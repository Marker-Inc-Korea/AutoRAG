# AutoRAG — Pi-Powered Librarian Agent

## Purpose

AutoRAG is an **over-powered librarian agent** for **document collections** — PDFs, wikis, notes, research papers, knowledge bases, and any unstructured text corpus. It is built on the [Pi framework](https://github.com/earendil-works/pi-mono).

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
Pi Built-in Tools          AutoRAG Extension Layer
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
| posix (agentdir virtual tree) | Active | Plain text, docs, config files — content search over the virtual layout |
| find (agentdir virtual tree) | Active | File discovery by name/glob over the virtual layout |
| Vector (semantic) | Planned | Dense documents, conceptual queries, "find similar to X" |
| BM25 (keyword) | Planned | Keyword-heavy search, term frequency ranking |
| Hybrid (vector+BM25) | Planned | Best-of-both fusion with score normalization |

The `RetrievalMethodRegistry` and `ResultMerger` are live: the agentdir `posix` method (`src/retrieval/methods/posix.ts`) is registered and routed through `ParallelRetriever` + `ResultMerger`. New methods (vector/BM25/hybrid) implement the `RetrievalMethod` interface and plug into the same pipeline.

## agentdir Integration

AutoRAG navigates document collections through [agentdir](https://github.com/NomaDamas/agentdir), an agent-optimized **read-only virtual folder** layer (consumed via the `@nomadamas/agentdir` Node binding). Source directories are mapped into a virtual tree at `.autorag/workspace`; the agent sees only virtual paths (e.g. `/docs/report.md`) — source filesystem paths are never exposed.

- **Tool surface replacement** — the Pi built-in `grep/find/read/ls` are replaced by agentdir virtual-path tools (`ls`=rglob, `read`=readBytes, `stat`=stat, `grep`=rglob+readBytes+regex), plus virtual ops `mv`/`cp`/`mkdir`/`rmdir` that rearrange the layout without touching source files. The active surface is closed via `setActiveTools(ACTIVE_TOOLS)`.
- **One grep core, two consumers** — `src/agentdir/grep-core.ts` backs both the `grep` tool and the `posix` `RetrievalMethod` (score = matchCount + 1/(1+depth)).
- **Change tracking** — `refresh()` runs on session start; opt-in `refreshWithHashVerification(true)` (the `autorag-refresh` command / `AutoRAGAgent.refresh(true)`) adds a SHA-256 pass that catches same-size/same-mtime content swaps (agentdir issue #2).
- **Organizer sub-agent** — a skeleton `organizer` agent (markdown definition) is delegated to via the `organize` tool (child-`pi` spawn, opt-in behind `AUTORAG_E2E_SPAWN=1`); it reorganizes the virtual layout with agentdir virtual ops. Concrete pipeline logic is deferred.

## Usage

### As Pi Extension (Interactive TUI)
```bash
pi --extension path/to/autorag/src/extension.ts
```

### As Library (Programmatic)
```typescript
import { AutoRAGAgent } from "@autorag/librarian";
import { getModel } from "@earendil-works/pi-ai";

const agent = new AutoRAGAgent({
  model: getModel("anthropic", "claude-sonnet-4-20250514"),
  searchPaths: ["/path/to/documents"],
});
const session = await agent.prompt("summarize the Q3 financial report");
agent.recordFeedbackByNumbers(session.sessionId, [1, 3], [2]);
```

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
2. Agent resolves numbers → internal mapping → source paths
3. Source paths → memory entries updated (useful/not_useful)
4. Memory informs future search strategy

## Files

| File | Role |
|------|------|
| `src/extension.ts` | Pi extension factory — registers tools, hooks events, injects system prompt |
| `src/agent/agent.ts` | AutoRAGAgent class for programmatic/library usage |
| `src/agent/system-prompt.ts` | System prompt builder (shared between extension and library modes) |
| `src/agent/parse-mapping.ts` | Internal mapping parser (number → source → method) |
| `src/memory/memory.ts` | Feedback persistence and method priority scoring |
| `src/memory/renderer.ts` | Memory context renderer for system prompt |
| `src/memory/check-memory-tool.ts` | check_memory tool (ToolDefinition + AgentTool) |
| `src/manifest/loader.ts` | YAML/JSON manifest loader for indexed data stores |
| `src/retrieval/types.ts` | Core retrieval type definitions |
| `src/retrieval/registry.ts` | Method registry for multi-method orchestration |
| `src/retrieval/merger.ts` | Cross-method result merging and deduplication |
| `src/retrieval/methods/posix.ts` | agentdir-backed `posix` RetrievalMethod (wraps the shared grep core) |
| `src/agentdir/workspace.ts` | agentdir Workspace lifecycle: open/init, mapping bootstrap, refresh |
| `src/agentdir/grep-core.ts` | Pure virtual-tree grep core shared by the grep tool and posix method |
| `src/agentdir/tools.ts` | agentdir virtual-path tools + `ACTIVE_TOOLS`/`AGENTDIR_TOOL_NAMES`/`SEARCH_TOOLS` |
| `src/agentdir/assert-no-source-path.ts` | Path-opacity guard (no source path in agent-facing output) |
| `src/organizer/agents.ts` | Organizer agent-definition discovery (frontmatter markdown) |
| `src/organizer/agents/organizer.md` | Bundled `organizer` sub-agent definition |
| `src/organizer/organize-tool.ts` | Spawn-tolerant `organize` delegation tool (child-`pi`) |
